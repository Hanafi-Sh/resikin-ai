# ResikIn Waste Classifier 🗑️🤖

Sistem klasifikasi gambar sampah berbasis **CLIP ViT-B/32** yang telah di-*fine-tune* secara *contrastive* untuk mendeteksi apakah sebuah foto berisi sampah/limbah atau bukan. Digunakan sebagai komponen AI dalam proyek **ResikIn** — platform pelaporan sampah berbasis web dan Telegram Bot untuk Kota Yogyakarta.

## 🎯 Problem Definition

Warga sering mengirimkan foto melalui bot Telegram atau website saat melaporkan masalah sampah. Diperlukan sistem AI yang dapat:
1. **Memvalidasi** apakah foto yang dikirim benar-benar berisi sampah (bukan selfie, makanan, dll)
2. **Mengkategorikan** jenis permasalahan sampah (TPS penuh, sampah liar, tidak terangkut)

---

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
│ (ViT-B/32)   │     │ (Transformer)│
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

---

## 🧠 AI Logic & Decision Support

Sistem ResikIn menggunakan dua pendekatan AI yang berbeda untuk membantu operasional:

### 1. Vision Analysis (CLIP Fine-tuned)
Digunakan untuk memvalidasi laporan warga di awal.
- **Tujuan**: Memastikan foto yang dikirim benar-benar berisi tumpukan sampah.
- **Efek**: Mengurangi beban verifikasi manual oleh Koordinator hingga **90%** karena laporan "sampah palsu" atau selfie akan langsung ditolak oleh Bot di awal.

### 2. Assignment Intelligence (DeepSeek LLM)
Digunakan saat Koordinator ingin menugaskan petugas melalui fitur **"Tanya AI"**.
- **Tujuan**: Merekomendasikan petugas terbaik berdasarkan beban kerja (*workload balancing*).
- **Cara Kerja**: AI membaca daftar tugas aktif setiap petugas dan mencocokkannya dengan kategori laporan. Jika seorang petugas sudah menangani banyak laporan, AI akan menyarankan petugas lain yang lebih senggang atau yang memiliki kualifikasi sesuai kategori (misal: petugas dengan spesialisasi pengangkutan tumpukan besar).

---

## 🚀 Instalasi & Persiapan (Local/Server)

Layanan ini dikembangkan menggunakan **FastAPI** untuk melayani request klasifikasi secara real-time.

### 1. Persiapan Environment
```bash
# Clone repository (jika belum)
git clone https://github.com/Hanafi-Sh/resikin-ai.git
cd resikin-waste-classifier

# Buat virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Menjalankan AI Service
```bash
# Menjalankan server pada port 8001
export MODEL_PATH="models/clip_waste_classifier.pth"
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

---

## 🌐 Dokumentasi API (Integrasi Server)

### 1. Validasi Gambar (Base64)
Endpoint utama untuk integrasi dengan Bot Telegram atau Web Apps.
- **Method**: `POST`
- **Endpoint**: `/api/ai/validate-image`
- **Body**: `{"image": "data:image/jpeg;base64,..."}`
- **Contoh Response (Waste)**:
  ```json
  {
    "success": true,
    "is_waste": true,
    "confidence": 0.9852,
    "suggested_category": "tps_penuh",
    "detail": {
      "waste_prob": 0.9852,
      "not_waste_prob": 0.0148,
      "subcategories": {
        "tps_penuh": 0.921,
        "sampah_liar": 0.052,
        "tidak_terangkut": 0.027
      }
    }
  }
  ```

### 2. Rekomendasi Petugas (Assignment)
Digunakan untuk mencari petugas yang paling cocok (beban kerja terendah) untuk menangani laporan tertentu.
- **Method**: `POST`
- **Endpoint**: `/api/ai/recommend-assignment`
- **Body**:
  ```json
  {
    "report_id": "uuid-laporan",
    "category": "sampah_liar",
    "petugas_list": [{"id": "p1", "name": "Agus"}, {"id": "p2", "name": "Budi"}],
    "active_assignments": [{"petugas_id": "p1", "status": "dalam_proses"}]
  }
  ```
- **Response**: `{"success": true, "recommended_petugas_id": "p2", "reason": "..."}`

### 3. Validasi Gambar (File Upload)
Gunakan ini untuk pengujian langsung tanpa perlu konversi base64.
- **Method**: `POST`
- **Endpoint**: `/api/ai/validate-image-file`
- **Form Data**: `file=@sampah.jpg`
- **Contoh Response (Bukan Sampah)**:
  ```json
  {
    "success": true,
    "is_waste": false,
    "confidence": 0.9912,
    "suggested_category": null,
    "detail": {
      "waste_prob": 0.0088,
      "not_waste_prob": 0.9912,
      "subcategories": {}
    }
  }
  ```

### 3. Predict Alias (DSAI Standard)
Alias untuk mematuhi format penugasan standar.
- **Method**: `POST`
- **Endpoint**: `/predict`
- **Body**: `{"image": "BASE64_STRING"}`

---

## 📮 Pengujian via Postman

Anda dapat menguji API ini secara visual menggunakan Postman:

### A. Menggunakan Base64
1. Pilih method **POST**.
2. URL: `http://43.156.249.230:8001/api/ai/validate-image`
3. Tab **Body** > **raw** > **JSON**.
4. Masukkan: `{"image": "iVBORw0KGgoAAAANSUhEUgAAAAE..."}`

### B. Menggunakan Upload File
1. Pilih method **POST**.
2. URL: `http://43.156.249.230:8001/api/ai/validate-image-file`
3. Tab **Body** > **form-data**.
4. Key: `file` (ubah tipe ke **File**), Value: Pilih gambar dari PC Anda.

---

## 💻 Contoh Integrasi (Python)

```python
import requests
import base64

def check_waste(image_path):
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    
    url = "http://43.156.249.230:8001/api/ai/validate-image"
    payload = {"image": img_b64}
    
    response = requests.post(url, json=payload)
    return response.json()
```

---

## 📊 Hasil Evaluasi

Model di-fine-tune selama 10 epoch di Google Colab (GPU T4) dan dievaluasi pada Test Set:

| Metric | Score |
|---|---|
| **Test Accuracy** | **96.43%** |
| Precision (Waste) | 92.3% |
| Recall (Waste) | 100.0% |
| F1-Score (Waste) | 96.0% |
| F1-Score (Not Waste) | 96.7% |

---

## 📝 Lisensi

Proyek ini dibuat untuk keperluan OTI Internship 2026.
