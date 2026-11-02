FROM ghcr.io/astral-sh/uv:0.12.0 AS uv
FROM python:3.11-slim
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY configs ./configs
RUN uv sync --frozen --no-dev
EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "ant26.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
