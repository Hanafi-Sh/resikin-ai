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

#### `POST /predict`

**Input (JSON):**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQ..."
}
```

**Output (JSON):**
```json
{
  "success": true,
  "is_waste": true,
  "confidence": 0.9234,
  "suggested_category": "sampah_liar",
  "detail": {
    "waste_prob": 0.9234,
    "not_waste_prob": 0.0766,
    "subcategories": {
      "tps_penuh": 0.2341,
      "sampah_liar": 0.5123,
      "tidak_terangkut": 0.1802,
      "bukan_sampah": 0.0734
    }
  }
}
```

#### Tes dengan cURL:
```bash
curl -X POST "http://localhost:8001/predict" \
  -H "Content-Type: application/json" \
  -d '{"image": "BASE64_IMAGE_STRING"}'
```

## 📊 Hasil Evaluasi

*(Akan diisi setelah training selesai)*

| Metric | Score |
|---|---|
| Accuracy | - |
| F1-Score (waste) | - |
| F1-Score (not_waste) | - |

## 🔧 Training

Training dilakukan di Google Colab (GPU T4) menggunakan contrastive fine-tuning:

```bash
python scripts/train.py --data_dir ./data --epochs 10 --batch_size 32 --lr 1e-5
```

## 📝 Lisensi

Proyek ini dibuat untuk keperluan OTI Internship 2026.
