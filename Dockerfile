FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY raglab ./raglab
COPY data ./data
# Embeddings are computed on first start and cached here; mount a volume to keep them.
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "raglab.api:app", "--host", "0.0.0.0", "--port", "8000"]
