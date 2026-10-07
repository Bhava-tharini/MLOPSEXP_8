FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --default-timeout=300 --retries=5 --no-cache-dir -r requirements.txt

COPY . .
RUN python -m src.preprocess && python -m src.feature_selection && python -m src.train

EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
