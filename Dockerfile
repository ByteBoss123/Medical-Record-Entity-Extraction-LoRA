FROM python:3.11-slim
WORKDIR /app
COPY serve/requirements.txt serve/requirements.txt
RUN pip install --no-cache-dir -r serve/requirements.txt
COPY mre.py ./
COPY serve/ serve/
# serve/model (crf.pkl, train_meta.json) is produced by serve/train_crf.py before the build
RUN test -f serve/model/crf.pkl && useradd -u 10001 app
USER 10001
EXPOSE 8000
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
