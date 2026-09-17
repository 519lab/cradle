# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable
ARG BAKE_EMBEDDINGS=1
# Bake the L2 cross-encoder reranker (features.l2_rerank, ~1.1GB) so a fresh
# container is self-contained and does not download it at startup warm-up. Set
# BAKE_RERANK=0 to skip (e.g. rerank disabled) and keep the image small.
ARG BAKE_RERANK=1
# Hugging Face token, available to the bake steps below so gated or
# rate-limited model downloads authenticate. Optional; unset = anonymous.
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
ENV FASTEMBED_CACHE_PATH=/opt/cradle/models/fastembed
RUN mkdir -p /opt/cradle/models/fastembed \
    && if [ "$BAKE_EMBEDDINGS" = "1" ]; then \
         /app/.venv/bin/python -c \
           "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5', cache_dir='/opt/cradle/models/fastembed')"; \
       fi \
    && if [ "$BAKE_RERANK" = "1" ]; then \
         /app/.venv/bin/python -c \
           "from fastembed.rerank.cross_encoder import TextCrossEncoder; TextCrossEncoder(model_name='BAAI/bge-reranker-base', cache_dir='/opt/cradle/models/fastembed')"; \
       fi

FROM python:3.12-slim-bookworm
RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin cradle \
    && mkdir -p /data /opt/cradle/models/fastembed \
    && chown -R cradle:cradle /data /opt/cradle
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY config /app/config
COPY --from=builder --chown=cradle:cradle /opt/cradle/models /opt/cradle/models
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod 755 /app/entrypoint.sh
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    CRADLE_SERVER__HOST=0.0.0.0 \
    CRADLE_DATA_DIR=/data \
    CRADLE_CONFIG=/app/config/cradle.yaml \
    FASTEMBED_CACHE_PATH=/opt/cradle/models/fastembed
USER root
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz')"
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "-m", "cradle"]
