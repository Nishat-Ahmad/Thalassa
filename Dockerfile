# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Speed up and harden pip in flaky networks
ENV PIP_DEFAULT_TIMEOUT=300 \
	PIP_PROGRESS_BAR=off \
	PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt /app/requirements.txt

# Pre-upgrade pip and retry install once on failure to mitigate timeouts
RUN python -m pip install --upgrade pip setuptools wheel \
	&& (pip install --no-cache-dir -r /app/requirements.txt \
		|| (echo "Retrying pip install after 15s..." && sleep 15 && pip install --no-cache-dir -r /app/requirements.txt))

COPY . /app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
