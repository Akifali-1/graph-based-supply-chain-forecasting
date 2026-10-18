FROM python:3.11-slim

WORKDIR /app

COPY requirements.lock /tmp/requirements.lock
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r /tmp/requirements.lock

COPY . .
ENV PYTHONPATH=/app

CMD ["python", "-m", "pytest", "-q"]
