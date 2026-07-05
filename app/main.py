import gc
import json
import logging
import os
import shutil
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

try:
    from openocr import OpenOCR
except ImportError as exc:  # pragma: no cover - startup guard for misconfigured images
    OpenOCR = None
    OPENOCR_IMPORT_ERROR = exc
else:
    OPENOCR_IMPORT_ERROR = None

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

TMP_ROOT = Path(os.getenv("OPENOCR_TMP_DIR", "/tmp"))
API_KEY = os.getenv("API_KEY")
MAX_UPLOAD_BYTES = int(os.getenv("OPENOCR_MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
UPLOAD_CHUNK_BYTES = 1024 * 1024
DOC_MAX_PARALLEL_BLOCKS = int(os.getenv("OPENOCR_DOC_MAX_PARALLEL_BLOCKS", "1"))
FILE_URL_TIMEOUT_SECONDS = float(os.getenv("OPENOCR_FILE_URL_TIMEOUT_SECONDS", "30"))

_engine_lock = threading.Lock()
_active_engine: Any | None = None
_active_engine_key: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    await run_in_threadpool(_warm_up_models)
    yield


app = FastAPI(title="openocr-service", lifespan=lifespan)


def _require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if not API_KEY:
        return

    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ocr")
async def ocr(
    file: UploadFile | None = File(default=None),
    file_url: str | None = Form(default=None, alias="fileUrl"),
    _: None = Depends(_require_api_key),
) -> dict[str, Any]:
    return await _process_input(file, file_url, "ocr")


@app.post("/doc")
async def doc(
    file: UploadFile | None = File(default=None),
    file_url: str | None = Form(default=None, alias="fileUrl"),
    _: None = Depends(_require_api_key),
) -> dict[str, Any]:
    return await _process_input(file, file_url, "doc")


async def _process_input(
    file: UploadFile | None,
    file_url: str | None,
    task: str,
) -> dict[str, Any]:
    if OPENOCR_IMPORT_ERROR is not None:
        raise HTTPException(
            status_code=500,
            detail=f"openocr-python is not installed correctly: {OPENOCR_IMPORT_ERROR}",
        )

    file_url = file_url.strip() if file_url else None
    if (file is None and not file_url) or (file is not None and file_url):
        raise HTTPException(status_code=400, detail="Send exactly one of file or fileUrl.")

    request_id = str(uuid.uuid4())
    work_dir = TMP_ROOT / f"openocr-{request_id}"
    output_dir = work_dir / "output"
    work_dir.mkdir(parents=True, exist_ok=False)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_name = file.filename if file is not None else _filename_from_url(file_url)
    input_path = work_dir / _safe_input_name(source_name, request_id)

    try:
        if file is not None:
            source = "file"
            await run_in_threadpool(_copy_upload, file.file, input_path)
        else:
            source = "fileUrl"
            await run_in_threadpool(_copy_file_url, file_url, input_path)

        result = await run_in_threadpool(_run_openocr, task, input_path, output_dir)
        return {
            "request_id": request_id,
            "source": source,
            "filename": source_name,
            "task": task,
            "result": _normalize_result(task, result),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if file is not None:
            await file.close()
        shutil.rmtree(work_dir, ignore_errors=True)


def _copy_upload(src: Any, dst_path: Path) -> None:
    total_bytes = 0

    with dst_path.open("wb") as dst:
        while chunk := src.read(UPLOAD_CHUNK_BYTES):
            total_bytes += len(chunk)
            if total_bytes > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"Upload too large. Max size is {MAX_UPLOAD_BYTES} bytes.",
                )
            dst.write(chunk)


def _copy_file_url(file_url: str, dst_path: Path) -> None:
    parsed_url = urlparse(file_url)
    if parsed_url.scheme not in {"http", "https"}:
        raise HTTPException(status_code=400, detail="fileUrl must use http or https.")

    request = Request(file_url, headers={"User-Agent": "openocr-service/1.0"})

    try:
        with urlopen(request, timeout=FILE_URL_TIMEOUT_SECONDS) as response:
            content_length = response.headers.get("Content-Length")
            try:
                content_length_bytes = int(content_length) if content_length else 0
            except ValueError:
                content_length_bytes = 0

            if content_length_bytes > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"fileUrl too large. Max size is {MAX_UPLOAD_BYTES} bytes.",
                )

            _copy_upload(response, dst_path)
    except HTTPException:
        raise
    except HTTPError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"fileUrl download failed with HTTP {exc.code}.",
        ) from exc
    except (TimeoutError, URLError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"fileUrl download failed: {exc}") from exc


def _filename_from_url(file_url: str | None) -> str | None:
    if not file_url:
        return None

    parsed_url = urlparse(file_url)
    filename = Path(unquote(parsed_url.path)).name
    return filename or None


def _safe_input_name(filename: str | None, request_id: str) -> str:
    suffix = Path(filename or "").suffix.lower()
    return f"{request_id}{suffix}"


def _run_openocr(task: str, input_path: Path, output_dir: Path) -> Any:
    with _engine_lock:
        engine = _get_engine(task)

        if task == "ocr":
            return engine(str(input_path), save_dir=str(output_dir), is_visualize=False)

        if task == "doc":
            return engine(str(input_path))

        raise ValueError(f"Unsupported task: {task}")


def _normalize_result(task: str, result: Any) -> Any:
    if task != "ocr":
        return _to_jsonable(result)

    if result is None:
        return {"items": [], "timings": []}

    if (
        isinstance(result, tuple)
        and len(result) == 2
    ):
        lines, timings = result
        if lines is None and timings is None:
            return {"items": [], "timings": []}

        if not lines:
            return {"items": [], "timings": _to_jsonable(timings) or []}

        return {
            "items": [_parse_ocr_line(line) for line in lines],
            "timings": _to_jsonable(timings) or [],
        }

    return _to_jsonable(result)


def _parse_ocr_line(line: Any) -> Any:
    if not isinstance(line, str) or "\t" not in line:
        return _to_jsonable(line)

    image_name, raw_result = line.split("\t", 1)
    return {
        "image_name": image_name,
        "ocr": _parse_json_string(raw_result),
    }


def _get_engine(task: str) -> Any:
    global _active_engine, _active_engine_key

    if _active_engine is not None and _active_engine_key == task:
        return _active_engine

    _active_engine = None
    _active_engine_key = None
    gc.collect()

    if task == "ocr":
        # infer_e2e.py compares use_gpu with `==` against the string 'false'.
        _active_engine = OpenOCR(task="ocr", mode="mobile", use_gpu="false")
    elif task == "doc":
        # infer_doc_onnx.py / infer_unirec_onnx.py compare use_gpu with `is False`,
        # so a string here silently falls through to GPU auto-detection instead of
        # forcing CPU.
        _active_engine = OpenOCR(
            task="doc",
            use_layout_detection=True,
            use_gpu=False,
            max_parallel_blocks=DOC_MAX_PARALLEL_BLOCKS,
        )
    else:
        raise ValueError(f"Unsupported task: {task}")

    _active_engine_key = task
    return _active_engine


def _warm_up_models() -> None:
    if OPENOCR_IMPORT_ERROR is not None:
        logger.warning("Skipping OpenOCR warm-up: %s", OPENOCR_IMPORT_ERROR)
        return

    for task in ("ocr", "doc"):
        try:
            _get_engine(task)
            logger.info("OpenOCR %s model warm-up completed", task)
        except Exception:
            logger.warning("OpenOCR %s model warm-up failed", task, exc_info=True)


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return _parse_json_string(value) if isinstance(value, str) else value

    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]

    if hasattr(value, "tolist"):
        return _to_jsonable(value.tolist())

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return str(value)


def _parse_json_string(value: str) -> Any:
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value
