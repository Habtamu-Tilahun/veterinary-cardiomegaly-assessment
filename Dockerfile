FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies (needed for OpenCV + Detectron2)
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Install Detectron2 (CPU build)
RUN pip install detectron2==0.6 \
    -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu117/torch2.0/index.html

# Copy project
COPY . .

# Expose Flask port
EXPOSE 5000

# Production server (better than app.run)
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "app:app"]