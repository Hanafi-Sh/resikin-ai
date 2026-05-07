# ResikIn Waste Classifier 🗑️🤖

Sistem klasifikasi gambar sampah berbasis **CLIP ViT-B/32** yang telah di-*fine-tune* secara *contrastive* untuk mendeteksi apakah sebuah foto berisi sampah/limbah atau bukan. Digunakan sebagai komponen AI dalam proyek **ResikIn** — platform pelaporan sampah berbasis web dan Telegram Bot untuk Kota Yogyakarta.

## 🎯 Problem Definition

Warga sering mengirimkan foto melalui bot Telegram atau website saat melaporkan masalah sampah. Diperlukan sistem AI yang dapat:
1. **Memvalidasi** apakah foto yang dikirim benar-benar berisi sampah (bukan selfie, makanan, dll)
2. **Mengkategorikan** jenis permasalahan sampah (TPS penuh, sampah liar, tidak terangkut)

## 🏗️ Arsitektur

```
┌──────────────┐     ┌──────────────┐
│  Image Input │     │  Text Labels │
│  (224x224)   │     │  (prompts)   │
└──────┬───────┘     └──────┬───────┘
       │                    │
       ▼                    ▼
┌──────────────┐     ┌──────────────┐
│ CLIP Vision  │     │  CLIP Text   │
│   Encoder    │     │   Encoder    │
│ (ViT-B/32)  │     │ (Transformer)│
│  Fine-tuned  │     │  Fine-tuned  │
└──────┬───────┘     └──────┬───────┘
       │                    │
       ▼                    ▼
┌──────────────┐     ┌──────────────┐
│   Visual     │     │    Text      │
│  Projection  │     │  Projection  │
└──────┬───────┘     └──────┬───────┘
       │                    │
       └────────┬───────────┘
                │
                ▼
        ┌───────────────┐
        │   Cosine      │
        │  Similarity   │
        │   + Softmax   │
        └───────┬───────┘
                │
                ▼
        ┌───────────────┐
        │  Prediction:  │
        │  waste /      │
        │  not_waste    │
        └───────────────┘
```

## 📁 Struktur Folder

```
resikin-waste-classifier/
├── README.md
├── report.pdf
├── requirements.txt
├── requirements-train.txt
├── data/
│   ├── raw/
│   ├── train/
│   ├── val/
│   └── test/
├── models/
│   └── clip_waste_classifier.pth
├── src/
│   ├── preprocessing/
│   │   └── prepare_dataset.py
│   ├── training/
│   ├── evaluation/
│   │   └── evaluate.py
│   └── utils/
├── app/
│   └── main.py
├── notebooks/
├── scripts/
│   └── train.py
```

## 🚀 Cara Menjalankan API

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Jalankan Server
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001
```

### 3. Endpoint API

#### `POST /predict` (Base64)
Mengirimkan gambar dalam format JSON base64. Cocok untuk integrasi bot.

**Input:** `{"image": "data:image/jpeg;base64,..."}`

#### `POST /predict-file` (File Upload)
Mengunggah file gambar secara langsung menggunakan `multipart/form-data`. **Sangat direkomendasikan untuk pengujian via Postman.**

**Body (form-data):**
- `file`: (Pilih file gambar Anda)

#### Tes dengan cURL:
```bash
# Menggunakan File Upload
curl -X POST "http://localhost:8001/predict-file" \
  -F "file=@path/ke/gambar.jpg"

# Menggunakan Base64
curl -X POST "http://localhost:8001/predict" \
  -H "Content-Type: application/json" \
  -d '{"image": "BASE64_STRING"}'
```

## 📊 Hasil Evaluasi

Model di-fine-tune selama 10 epoch di Google Colab (GPU T4) dan dievaluasi pada Test Set:

| Metric | Score |
|---|---|
| **Test Accuracy** | **96.43%** |
| Precision (Waste) | 92.3% |
| Recall (Waste) | 100.0% |
| F1-Score (Waste) | 96.0% |
| F1-Score (Not Waste) | 96.7% |

## 🔧 Training

Training dilakukan di Google Colab (GPU T4) menggunakan contrastive fine-tuning:

```bash
python scripts/train.py --data_dir ./data --epochs 10 --batch_size 32 --lr 1e-5
```

## 📝 Lisensi

Proyek ini dibuat untuk keperluan OTI Internship 2026.
