# Docker image for the free Hugging Face Space (CPU basic: 2 vCPU, 16 GB RAM).
# The index is built ONCE while the image builds, so the running app starts fast.
FROM python:3.11-slim

# Hugging Face runs the container as user 1000, so build everything as that user.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface
WORKDIR /home/user/app

# CPU-only PyTorch first: much smaller than the default GPU build.
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user ingest.py rag.py space_app.py ./

# Download the arXiv abstracts, chunk, embed and store them inside the image.
# This also downloads the embedding model into the image cache.
RUN python ingest.py

EXPOSE 7860
CMD ["uvicorn", "space_app:app", "--host", "0.0.0.0", "--port", "7860"]
