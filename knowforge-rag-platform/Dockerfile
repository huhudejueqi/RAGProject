# syntax=docker/dockerfile:1.7
ARG APP_BASE_IMAGE=localhost/knowforge-rag-platform-base:py312
FROM ${APP_BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV TOKENIZERS_PARALLELISM=false
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV GIT_PYTHON_REFRESH=quiet

WORKDIR /app

COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt \
    && python -m pip check \
    && python -c "import fastapi, uvicorn, langchain, langchain_milvus, pymilvus, sentence_transformers, transformers, safetensors, torch, pandas, fitz, docx, pptx, docling, ragas, markdown, mkdocs, redis; print('runtime dependencies ok')"

# The reusable base image may contain files from an older workspace snapshot.
# Clear application files before copying this release so removed V1 paths do not
# survive from a lower image layer.
RUN find /app -mindepth 1 -maxdepth 1 ! -name requirements.txt -exec rm -rf {} +

COPY . .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
