# openocr-service

FastAPI wrapper for `openocr-python==0.1.5` with CPU-only runtime.

## Endpoints

- `GET /health`
- `POST /ocr`: multipart field `file` or `fileUrl`, creates an OCR job
- `POST /doc`: multipart field `file` or `fileUrl`, creates a document parsing job
- `GET /jobs/{job_id}`: polls job status/result

Uploads are copied into `/tmp/openocr-jobs/<job_id>` and removed after processing.
Job results are kept in memory for up to 24 hours after completion.
Restarting the container clears in-memory jobs and results.
The service warms up OpenOCR models at startup and stores downloaded models in
`/root/.cache/openocr`.

## Configuration

- `API_KEY`: optional; when set, `/ocr`, `/doc`, and `/jobs/{job_id}` require header `X-API-Key`
- `HF_HOME=/root/.cache/openocr`: model/cache directory
- `OMP_NUM_THREADS=4`: CPU thread limit
- `OPENOCR_DOC_MAX_PARALLEL_BLOCKS=1`: keeps document parsing memory usage lower
- `OPENOCR_FILE_URL_TIMEOUT_SECONDS=30`: timeout for downloading `fileUrl`
- `OPENOCR_JOB_RESULT_TTL_SECONDS=86400`: result retention after job completion
- `OPENOCR_MAX_UPLOAD_BYTES=26214400`: max upload size, default 25 MB

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

## Bruno

Open `bruno/openocr-service` in Bruno and select the `Local` environment.

## Test

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/ocr \
  -F "file=@/path/to/image.jpg"

curl http://localhost:8000/jobs/<job_id>

curl -X POST http://localhost:8000/ocr \
  -F "fileUrl=https://example.com/image.jpg"

curl -X POST http://localhost:8000/doc \
  -F "file=@/path/to/document.pdf"

curl -X POST http://localhost:8000/doc \
  -F "fileUrl=https://example.com/document.pdf"
```

When `API_KEY` is set:

```bash
curl -X POST http://localhost:8000/ocr \
  -H "X-API-Key: your-api-key" \
  -F "file=@/path/to/image.jpg"

curl http://localhost:8000/jobs/<job_id> \
  -H "X-API-Key: your-api-key"
```
