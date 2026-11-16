# The demo: a Python server that writes control tracks, and a static page that plays them.
FROM python:3.13-slim

RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface
WORKDIR /home/user/app

# the model runs on the Space's two cores: PyTorch without CUDA
COPY --chown=user requirements.txt ./
COPY --chown=user wheels/ ./wheels/
RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt wheels/*.whl

COPY --chown=user . .
USER user
EXPOSE 7860
CMD ["python", "app.py"]
