# Gunakan image dasar Python
FROM python:3.8-slim

# Set working directory di dalam container
WORKDIR /app

# Salin file requirements.txt ke dalam container
COPY requirements.txt /app/

# Install dependensi
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file aplikasi ke dalam container
COPY . /app/

# Instal ekskripsi tambahan untuk Detectron2 (sesuai sistem operasi)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && pip install 'detectron2'  # Jika tidak ada di requirements.txt

# Tentukan port yang digunakan oleh aplikasi Flask
EXPOSE 5000

# Jalankan aplikasi Flask
CMD ["python", "app.py"]
