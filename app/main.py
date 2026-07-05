import gc
import json
import logging
import os
import shutil
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
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
JOB_ROOT = Path(os.getenv("OPENOCR_JOB_DIR", "/tmp/openocr-jobs"))
API_KEY = os.getenv("API_KEY")
MAX_UPLOAD_BYTES = int(os.getenv("OPENOCR_MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
UPLOAD_CHUNK_BYTES = 1024 * 1024
DOC_MAX_PARALLEL_BLOCKS = int(os.getenv("OPENOCR_DOC_MAX_PARALLEL_BLOCKS", "1"))
FILE_URL_TIMEOUT_SECONDS = float(os.getenv("OPENOCR_FILE_URL_TIMEOUT_SECONDS", "30"))
JOB_RESULT_TTL_SECONDS = int(os.getenv("OPENOCR_JOB_RESULT_TTL_SECONDS", str(24 * 60 * 60)))

_engine_lock = threading.Lock()
_active_engine: Any | None = None
_active_engine_key: str | None = None
_jobs_lock = threading.Lock()
_jobs: dict[str, dict[str, Any]] = {}
_cleanup_stop_event = threading.Event()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await run_in_threadpool(_warm_up_models)
    JOB_ROOT.mkdir(parents=True, exist_ok=True)
    _cleanup_stop_event.clear()
    cleanup_thread = threading.Thread(target=_cleanup_expired_jobs_loop, daemon=True)
    cleanup_thread.start()
    yield
    _cleanup_stop_event.set()
    cleanup_thread.join(timeout=5)


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
    return await _create_job(file, file_url, "ocr")


@app.post("/doc")
async def doc(
    file: UploadFile | None = File(default=None),
    file_url: str | None = Form(default=None, alias="fileUrl"),
    _: None = Depends(_require_api_key),
) -> dict[str, Any]:
    return await _create_job(file, file_url, "doc")


@app.get("/jobs/{job_id}")
def get_job(
    job_id: str,
    _: None = Depends(_require_api_key),
) -> dict[str, Any]:
    _cleanup_expired_jobs()

    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")

        return _public_job(job)


async def _create_job(
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

    if file_url:
        _validate_file_url(file_url)

    job_id = str(uuid.uuid4())
    work_dir = JOB_ROOT / job_id
    output_dir = work_dir / "output"
    work_dir.mkdir(parents=True, exist_ok=False)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_name = file.filename if file is not None else _filename_from_url(file_url)
    input_path = work_dir / _safe_input_name(source_name, job_id)
    source = "file" if file is not None else "fileUrl"

    try:
        if file is not None:
            await run_in_threadpool(_copy_upload, file.file, input_path)

        job = _new_job(job_id, task, source, source_name, file_url, work_dir)
        with _jobs_lock:
            _jobs[job_id] = job

        _log_job_progress(job_id, task, "queued")
        response = _public_job(job)
        thread = threading.Thread(
            target=_run_job,
            args=(job_id, task, input_path, output_dir, file_url),
            daemon=True,
        )
        thread.start()

        return response
    except HTTPException:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if file is not None:
            await file.close()


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
    _validate_file_url(file_url)
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


def _validate_file_url(file_url: str) -> None:
    parsed_url = urlparse(file_url)
    if parsed_url.scheme not in {"http", "https"}:
        raise HTTPException(status_code=400, detail="fileUrl must use http or https.")


def _filename_from_url(file_url: str | None) -> str | None:
    if not file_url:
        return None

    parsed_url = urlparse(file_url)
    filename = Path(unquote(parsed_url.path)).name
    return filename or None


def _safe_input_name(filename: str | None, request_id: str) -> str:
    suffix = Path(filename or "").suffix.lower()
    return f"{request_id}{suffix}"


def _new_job(
    job_id: str,
    task: str,
    source: str,
    filename: str | None,
    file_url: str | None,
    work_dir: Path,
) -> dict[str, Any]:
    now = _utc_now()
    return {
        "job_id": job_id,
        "status": "queued",
        "status_url": f"/jobs/{job_id}",
        "task": task,
        "source": source,
        "filename": filename,
        "file_url": file_url,
        "created_at": _format_dt(now),
        "started_at": None,
        "completed_at": None,
        "expires_at": None,
        "result": None,
        "error": None,
        "_work_dir": str(work_dir),
        "_expires_at": None,
    }


def _run_job(
    job_id: str,
    task: str,
    input_path: Path,
    output_dir: Path,
    file_url: str | None,
) -> None:
    _update_job(job_id, status="running", started_at=_format_dt(_utc_now()))
    _log_job_progress(job_id, task, "running")

    try:
        if file_url:
            _copy_file_url(file_url, input_path)

        result = _run_openocr(task, input_path, output_dir)
        _finish_job(
            job_id,
            status="succeeded",
            result=_normalize_result(task, result),
            error=None,
        )
        _log_job_progress(job_id, task, "succeeded")
    except HTTPException as exc:
        _finish_job(
            job_id,
            status="failed",
            result=None,
            error={"status_code": exc.status_code, "detail": exc.detail},
        )
        _log_job_progress(job_id, task, "failed")
    except Exception as exc:
        _finish_job(
            job_id,
            status="failed",
            result=None,
            error={"status_code": 500, "detail": str(exc)},
        )
        _log_job_progress(job_id, task, "failed")
    finally:
        with _jobs_lock:
            job = _jobs.get(job_id)
            work_dir = Path(job["_work_dir"]) if job else None

        if work_dir is not None:
            shutil.rmtree(work_dir, ignore_errors=True)


def _update_job(job_id: str, **updates: Any) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is not None:
            job.update(updates)


def _finish_job(
    job_id: str,
    status: str,
    result: Any,
    error: dict[str, Any] | None,
) -> None:
    now = _utc_now()
    expires_at = now + timedelta(seconds=JOB_RESULT_TTL_SECONDS)
    _update_job(
        job_id,
        status=status,
        completed_at=_format_dt(now),
        expires_at=_format_dt(expires_at),
        result=result,
        error=error,
        _expires_at=expires_at,
    )


def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "status_url": job["status_url"],
        "task": job["task"],
        "source": job["source"],
        "filename": job["filename"],
        "created_at": job["created_at"],
        "started_at": job["started_at"],
        "completed_at": job["completed_at"],
        "expires_at": job["expires_at"],
        "result": job["result"] if job["status"] == "succeeded" else None,
        "error": job["error"],
    }


def _cleanup_expired_jobs_loop() -> None:
    while not _cleanup_stop_event.wait(300):
        _cleanup_expired_jobs()


def _cleanup_expired_jobs() -> None:
    now = _utc_now()
    expired_jobs: list[dict[str, Any]] = []

    with _jobs_lock:
        for job_id, job in list(_jobs.items()):
            expires_at = job.get("_expires_at")
            if expires_at is not None and expires_at <= now:
                expired_jobs.append(_jobs.pop(job_id))

    for job in expired_jobs:
        shutil.rmtree(job["_work_dir"], ignore_errors=True)
        _log_job_progress(job["job_id"], job["task"], "expired")


def _log_job_progress(job_id: str, task: str, status: str) -> None:
    logger.info("job_id=%s task=%s status=%s", job_id, task, status)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _format_dt(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


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
        except Exception as exc:
            logger.warning("OpenOCR %s model warm-up failed: %s", task, exc)


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
