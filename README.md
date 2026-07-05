# openocr-service

FastAPI wrapper for `openocr-python==0.1.5` with CPU-only runtime.

## Endpoints

- `GET /health`
- `POST /extract/file`: multipart field `file`, creates an extraction job
- `POST /extract/url?url=...`: query parameter `url`, creates an extraction job
- `GET /jobs/{job_id}`: polls job status/result

Uploads are copied into `/tmp/openocr-jobs/<job_id>` and removed after processing.
Job results are kept in memory for up to 24 hours after completion.
Successful job results include a `markdown` field alongside the structured data.
Restarting the container clears in-memory jobs and results.
Word extraction supports `.docx` and legacy `.doc`; Excel extraction supports `.xlsx/.xlsm`.
PDF/image documents are handled by OpenOCR. YouTube URLs use available transcripts/captions.
The service warms up OpenOCR models at startup and stores downloaded models in
`/root/.cache/openocr`.

## Configuration

- `API_KEY`: optional; when set, `/extract/file`, `/extract/url`, and `/jobs/{job_id}` require header `X-API-Key`
- `HF_HOME=/root/.cache/openocr`: model/cache directory
- `OMP_NUM_THREADS=4`: CPU thread limit
- `OPENOCR_DOC_MAX_PARALLEL_BLOCKS=1`: keeps document parsing memory usage lower
- `OPENOCR_FILE_URL_TIMEOUT_SECONDS=30`: timeout for downloading remote URLs
- `OPENOCR_JOB_RESULT_TTL_SECONDS=86400`: result retention after job completion
- `OPENOCR_MAX_UPLOAD_BYTES=26214400`: max upload size, default 25 MB

## Response Format

`POST /extract/file` and `POST /extract/url` return a job immediately:

```json
{
  "job_id": "uuid",
  "status": "queued",
  "status_url": "/jobs/uuid",
  "task": "ocr|doc|word|csv|excel|youtube",
  "source": "file|fileUrl|youtubeUrl",
  "filename": "input filename or detected id",
  "created_at": "2026-07-05T11:08:21.897597Z",
  "started_at": null,
  "completed_at": null,
  "expires_at": null,
  "result": null,
  "error": null
}
```

Poll `GET /jobs/{job_id}` until `status` is terminal:

- `queued`: job is created but worker has not started.
- `running`: worker is processing the input.
- `succeeded`: extraction completed; `result` is populated.
- `failed`: extraction failed; `error` is populated.

Common fields:

- `job_id`: UUID for polling.
- `status_url`: relative polling URL for this job.
- `task`: detected extractor. `ocr` is image OCR, `doc` is PDF/OpenOCR document parsing, `word` is `.doc/.docx`, `csv` is CSV, `excel` is `.xlsx/.xlsm`, `youtube` is transcript extraction.
- `source`: input source. `file` means multipart upload, `fileUrl` means remote file URL, `youtubeUrl` means YouTube URL.
- `filename`: uploaded filename, remote filename, or YouTube video id.
- `created_at`, `started_at`, `completed_at`, `expires_at`: UTC ISO-8601 timestamps. `expires_at` is set after completion.
- `result`: `null` until `succeeded`; then contains structured data plus `markdown`.
- `error`: `null` unless `failed`; then contains `{ "status_code": number, "detail": string }`.

Successful `result` always includes:

- `markdown`: extracted content in Markdown format, suitable for LLM/RAG ingestion or plain preview.

Result shape by task:

- `ocr`: `{ "items": [...], "timings": [...], "markdown": "..." }`
  - `items[].image_name`: image filename/page item.
  - `items[].ocr`: raw OpenOCR OCR payload.
- `doc`: OpenOCR document parsing payload plus `markdown`.
  - Used for PDF and document images.
  - Shape depends on OpenOCR output; unsupported fields are preserved as JSON-compatible values.
- `word`: `{ "paragraphs": [...], "tables": [...], "paragraph_count": n, "table_count": n, "markdown": "..." }`
  - `paragraphs`: non-empty paragraph text from `.doc/.docx`.
  - `tables`: array of tables; each table is an array of row arrays.
- `csv`: `{ "columns": [...], "rows": [...], "row_count": n, "markdown": "..." }`
  - `columns`: first CSV row.
  - `rows`: remaining CSV rows.
- `excel`: `{ "sheets": [...], "sheet_count": n, "markdown": "..." }`
  - `sheets[].name`: worksheet name.
  - `sheets[].columns`: first non-empty row.
  - `sheets[].rows`: remaining rows.
  - `sheets[].row_count`: number of data rows.
- `youtube`: `{ "video_id": "...", "segments": [...], "segment_count": n, "markdown": "..." }`
  - `segments[].text`: transcript text.
  - `segments[].start`: start time in seconds.
  - `segments[].duration`: segment duration in seconds.

## Build

```bash
docker build -t openocr-service .
```

## Run

```bash
docker volume create openocr-cache

docker run --rm -p 8000:8000 --cpus 4 --memory 4096m \
  -v openocr-cache:/root/.cache/openocr \
  openocr-service
```

Or with Docker Compose / Dokploy:

```bash
docker compose up --build
```

`docker-compose.yml` defines a named volume `openocr-cache` so Dokploy
redeploys do not re-download OpenOCR models.

## Local Dev

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p .cache/openocr

HF_HOME="$(pwd)/.cache/openocr" \
OMP_NUM_THREADS=4 \
OPENOCR_DOC_MAX_PARALLEL_BLOCKS=1 \
OPENOCR_FILE_URL_TIMEOUT_SECONDS=30 \
OPENOCR_JOB_RESULT_TTL_SECONDS=86400 \
OPENOCR_MAX_UPLOAD_BYTES=26214400 \
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Legacy `.doc` extraction uses `antiword`; install it locally if you need `.doc` support outside Docker.

## Bruno

Open `bruno/openocr-service` in Bruno and select the `Local` environment.

## Test

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/extract/file \
  -F "file=@/path/to/file.pdf"

curl http://localhost:8000/jobs/<job_id>

curl -X POST "http://localhost:8000/extract/url?url=https://example.com/file.pdf"

curl -X POST "http://localhost:8000/extract/url?url=https://example.com/document.docx"

curl -X POST "http://localhost:8000/extract/url?url=https://example.com/data.csv"

curl -X POST "http://localhost:8000/extract/url?url=https://example.com/workbook.xlsx"

curl -X POST "http://localhost:8000/extract/url?url=https://www.youtube.com/watch?v=VIDEO_ID"
```

When `API_KEY` is set:

```bash
curl -X POST http://localhost:8000/extract/file \
  -H "X-API-Key: your-api-key" \
  -F "file=@/path/to/file.pdf"

curl http://localhost:8000/jobs/<job_id> \
  -H "X-API-Key: your-api-key"
```

Build docker image

```bash
docker buildx build \
    --platform linux/amd64 \
    -t jasonvnit/openocr-service:latest \
    --push \
    .
```
