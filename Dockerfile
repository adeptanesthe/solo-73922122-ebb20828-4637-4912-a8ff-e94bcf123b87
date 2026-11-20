# CPU-only image. PyTorch is installed from the official CPU wheel index so the
# container never pulls CUDA libraries.
FROM python:3.11-slim

ARG TORCH_VERSION=2.15.0

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

WORKDIR /srv

COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        "torch==${TORCH_VERSION}" \
    && pip install -r requirements.txt

COPY app ./app
COPY tests ./tests

RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /srv
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
