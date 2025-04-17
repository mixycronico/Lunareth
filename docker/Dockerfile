FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY configs/ ./configs/
COPY main.py .

ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=development

CMD ["python", "main.py"]