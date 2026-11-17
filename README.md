# EmoReview — GA Feature Selection & Emotion Classification

Aplikasi web satu-platform berbasis **Python + Streamlit** untuk:

1. Mengunggah dataset ulasan produk berlabel.
2. Menjalankan Genetic Algorithm untuk memilih subset fitur TF-IDF.
3. Melatih dan mengevaluasi Linear SVM menggunakan fitur terpilih.
4. Mengklasifikasikan ulasan baru secara individual atau batch.
5. Membandingkan baseline dengan GA-selected model.

Tidak ada frontend dan backend terpisah. Seluruh UI, GA, preprocessing, dan inferensi berjalan dari aplikasi Python yang sama.

## Struktur project

```text
emoreview_streamlit/
├── app.py
├── model_utils.py
├── ml_pipeline.py
├── requirements.txt
├── README.md
├── artifacts/
│   ├── final_models.joblib
│   ├── final_evaluation_metadata.json
│   ├── final_test_metrics.csv
│   ├── baseline_confusion_matrix.csv
│   ├── ga_confusion_matrix.csv
│   ├── baseline_classification_report.csv
│   ├── ga_classification_report.csv
│   └── final_selected_features.csv
├── pages/
│   ├── 1_GA_Optimization.py
│   ├── 2_Model_Evaluation.py
│   └── 3_Emotion_Classification.py
└── outputs/
```

## Menjalankan aplikasi

Gunakan Python 3.13. Model bawaan diekspor menggunakan scikit-learn 1.6.1, NumPy 2.4.6, pandas 2.2.3, dan joblib 1.5.3. Versi scikit-learn tidak dipatok di requirements.txt; aplikasi akan mencoba memuat model dengan versi yang terpasang dan menampilkan peringatan bila versinya berbeda.

### Windows PowerShell

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

### macOS / Linux

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Browser akan membuka aplikasi Streamlit pada alamat lokal yang ditampilkan di terminal (umumnya `http://localhost:8501`).

## Isi aplikasi

### 1. Dashboard

Menampilkan ringkasan model bawaan dari eksperimen PRDECT-ID: jumlah fitur, tingkat reduksi, dan perbandingan metrik test baseline dengan GA.

### 2. Dataset & GA Optimization

- Menerima CSV dengan satu kolom teks dan satu kolom label.
- Default nama kolom: `Customer Review` dan `Emotion`.
- Label yang didukung: `Anger`, `Fear`, `Happy`, `Love`, dan `Sadness`.
- Duplicate key menggunakan casefold, penghapusan URL, penggantian tanda baca Unicode dengan spasi, dan normalisasi whitespace.
- Kelompok teks dengan label konflik dibuang seluruhnya; duplikat dengan label sama disisakan satu.
- Dataset dibagi secara stratified 70:15:15.
- TF-IDF unigram + bigram, `min_df=1`, di-fit hanya pada training.
- GA menggunakan konfigurasi awal dari hasil confirmation: population 40, crossover 0.9, mutation 100/L, tournament k=3, elitism 1, maksimal 30 generasi, patience 5, `min_delta=0.001`, dan penalti ukuran 0.02.
- Pengguna dapat mengubah beberapa parameter untuk eksperimen baru. Hasil eksperimen baru tidak menggantikan artefak model penelitian bawaan.
- Setelah GA selesai, baseline dan GA-selected SVM dilatih ulang pada train + validation, lalu dievaluasi pada test yang ditahan.
- Model hasil optimasi dapat diunduh sebagai `.joblib`.

### 3. Model Evaluation

Menampilkan metrik test model bawaan v10 atau model hasil optimasi terbaru dalam sesi aplikasi, termasuk confusion matrix dan classification report.

### 4. Emotion Classification

- Single review: klasifikasi satu ulasan dan bandingkan prediksi baseline dengan GA.
- Batch CSV: klasifikasi sekumpulan ulasan sekaligus, tampilkan distribusi prediksi, dan unduh hasil CSV.
- Hasil batch menyimpan prediksi baseline dan GA secara terpisah. Kolom `Selected_Model_Prediction`, grafik, dan unduhan mengikuti pilihan model yang sedang aktif.
- Mengganti file CSV atau kolom ulasan membatalkan hasil batch sebelumnya agar hasil dari input lama tidak tertukar dengan input baru.

## Catatan metodologis

- Hasil evaluasi bawaan menggunakan test split yang sama dengan evaluasi v9 sebelumnya. Hasil v10 adalah evaluasi lanjutan, bukan holdout independen yang sepenuhnya baru. Metadata menyatakan pemilihan konfigurasi dan fitur tidak menggunakan test set.

- **GA tidak dijalankan setiap kali pengguna meminta prediksi.** GA berjalan hanya ketika pengguna membuka halaman optimasi dan menekan tombol Run Genetic Algorithm. Hasil model disimpan dan kemudian dapat digunakan berulang kali.
- Mode klasifikasi menggunakan model tersimpan dari eksperimen v10 secara langsung.
- Dalam mode optimasi, test set hanya digunakan setelah kromosom terbaik dipilih berdasarkan validation.
- Vocabulary TF-IDF dikunci saat refit train + validation agar indeks kromosom tetap konsisten.
- Metrik test model bawaan adalah hasil eksperimen PRDECT-ID sebelumnya, bukan jaminan performa untuk dataset lain.
- Probabilitas kelas tidak ditampilkan karena `LinearSVC` bawaan belum dikalibrasi menjadi probabilitas.
- Model `.joblib` menggunakan mekanisme berbasis pickle. Jangan memuat model dari sumber yang tidak dipercaya.
- Model scikit-learn yang disimpan pada satu versi tidak memiliki jaminan kompatibilitas dengan versi lain. Peringatan versi pada aplikasi bersifat nonfatal; bila pemuatan atau prediksi gagal/tidak konsisten, gunakan lingkungan versi asal atau latih ulang model pada lingkungan target.

## Artefak model bawaan

Delapan artefak di folder `artifacts/` berasal dari final evaluation v10. Berkas yang digunakan aplikasi meliputi `final_models.joblib`, `final_evaluation_metadata.json`, `final_test_metrics.csv`, `final_selected_features.csv`, dua classification report, dan dua confusion matrix. `artifacts/final_models.joblib` berasal dari final evaluation v10. Metadata model mencatat scikit-learn 1.6.1, NumPy 2.4.6, dan pandas 2.2.3. Versi scikit-learn sengaja tidak dipatok di `requirements.txt`. Jika versi yang terpasang berbeda, aplikasi tidak otomatis dihentikan: aplikasi akan mencoba memuat model dan menampilkan peringatan kompatibilitas. Keberhasilan memuat tidak berarti kompatibilitas atau kesamaan hasil dijamin. Untuk reproduksi yang paling konsisten, gunakan scikit-learn 1.6.1.

## Batasan prototipe

- Aplikasi lokal untuk demonstrasi dan penggunaan terkontrol; belum memiliki autentikasi, database, antrean job, atau deployment produksi.
- Proses GA berjalan sinkron pada sesi Streamlit. Dataset yang besar dapat membuat proses berjalan lama.
- Belum terintegrasi langsung dengan API marketplace; pengguna menyediakan file ulasan sendiri.
- Model penelitian dilatih pada domain ulasan PRDECT-ID. Generalisasi ke domain lain belum dievaluasi.
