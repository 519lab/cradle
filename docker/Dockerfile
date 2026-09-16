FROM python:3.12-slim

RUN useradd --create-home --uid 1000 cradle
WORKDIR /app
COPY pyproject.toml README.md LICENSE /app/
COPY src /app/src
COPY config /app/config
RUN pip install --no-cache-dir . \
    && mkdir -p /data \
    && chown cradle:cradle /data
USER cradle
ENV CRADLE_SERVER__HOST=0.0.0.0
ENV CRADLE_DATA_DIR=/data
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz')"
CMD ["python", "-m", "cradle"]
