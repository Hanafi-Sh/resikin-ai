# Gunakan image Python ringan
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
# HuggingFace Spaces berjalan di port 7860
ENV PORT=7860

# Set working directory
WORKDIR /app

# Install dependencies sistem yang diperlukan (opsional, untuk opencv/Pillow dll)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Salin dan install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Buat folder struktur
RUN mkdir -p /app/models /app/app

# Salin file model dan kode aplikasi
COPY models/clip_waste_classifier/clip_waste_classifier.pth /app/models/
COPY app/ /app/app/

# Expose port untuk HuggingFace
EXPOSE 7860

# Jalankan server FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
