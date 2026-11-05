# Predictive Risk Modeling with Statistical and Machine Learning Approaches

**Wawan Saputra | Statistics & Data Science Portfolio | 2026**

Project pertama dari roadmap portfolio: membandingkan model statistik dan machine learning untuk klasifikasi diagnosis berdasarkan pengukuran citra. Seluruh angka hasil berasal dari eksekusi kode, bukan contoh buatan.

## Research / Business Question
Seberapa baik logistic regression dan model berbasis pohon membedakan sampel malignant dan benign pada data yang belum digunakan untuk pelatihan? Prediktor mana yang paling memengaruhi performa model?

## Dataset
Wisconsin Diagnostic Breast Cancer, tersedia lokal melalui `sklearn.datasets.load_breast_cancer`: 569 sampel, 30 fitur numerik hasil pengukuran citra fine-needle aspirate. Target dikodekan ulang: **1 = malignant**, 0 = benign. Data tidak mencakup fitur kategorik. Dokumentasi sumber, referensi dan deskripsi variabel disalin dari paket ke [DATASET.md](data/DATASET.md). CSV raw merupakan snapshot ekspor dataset bawaan, tanpa unduhan tambahan.

## Methods
- Logistic Regression sebagai baseline statistik.
- Decision Tree, Random Forest, Gradient Boosting dengan konfigurasi tetap.
- Split stratified 60% train, 20% validation, 20% test; seed 42.
- Median imputation dan standardization di dalam setiap pipeline/fold.
- Lima fold CV pada training; model dipilih menggunakan rata-rata ROC-AUC CV.
- Threshold dipilih berdasarkan F1 validation; model tidak di-refit sesudah pemilihan threshold.
- Test hanya mengevaluasi model dan threshold final. Tabel perbandingan berisi CV dan validation; tidak membandingkan kandidat berdasarkan test.
- ROC-AUC, average precision, precision, recall, F1, confusion matrix, Brier score dan kurva calibration.
- Permutation importance pada validation (20 pengulangan) dan partial dependence sebagai interpretasi model.

## Reproducibility
Jalankan di folder project dengan Python 3.11 atau lebih baru:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.train
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`requirements-lock.txt` mencatat versi paket inti yang benar-benar digunakan. Untuk mengulang lingkungan analisis tersebut gunakan `pip install -r requirements-lock.txt`; instal JupyterLab terpisah bila ingin membuka notebook. Versi Python tercatat dalam `outputs/environment.json`. Eksekusi ulang menimpa output analisis di dalam project.

## Workflow
1. Audit data dan ekspor snapshot, statistik deskriptif serta distribusi kelas.
2. Bentuk partisi yang saling terpisah.
3. Latih empat pipeline dengan CV training-only.
4. Pilih model dan threshold tanpa menggunakan test.
5. Evaluasi final, simpan prediksi, model, tabel dan visualisasi.
6. Hasilkan executive summary dan dua bullet CV dari hasil aktual.

EDA seluruh dataset hanya deskriptif; tidak digunakan untuk memilih fitur atau mengubah konfigurasi model. Semua 30 fitur dipertahankan, tanpa feature engineering berbasis target.

## Key Results
Lihat [executive summary](outputs/executive_summary.md), [model comparison](outputs/tables/model_comparison.csv), dan [test metrics](outputs/test_metrics.json). Tabel perbandingan menyertakan variasi antar-fold; ini bukan confidence interval.

## Visualizations
![Test ROC](outputs/figures/test_roc.png)
![Permutation importance](outputs/figures/permutation_importance.png)

Output tambahan: distribusi kelas, korelasi fitur, confusion matrix, calibration, dan partial dependence.

## Interpretation
ROC-AUC mengukur ranking probabilitas, sedangkan precision/recall/F1 bergantung pada threshold. Brier score yang lebih rendah lebih baik. Kurva calibration dengan test kecil hanya diagnostik awal. Penurunan ROC-AUC pada permutation importance menunjukkan ketergantungan model; fitur berkorelasi dapat saling menggantikan. Partial dependence tidak menunjukkan efek kausal.

## Limitations and Future Work
Dataset historis dari satu sumber dan ukuran test kecil membatasi generalisasi. Tidak tersedia validasi eksternal, interval ketidakpastian performa, audit fairness, atau identitas kelompok pasien untuk group splitting. Klasifikasi pengukuran citra ini tidak sama dengan prediksi risiko populasi atau validasi alat skrining. Model edukatif ini belum divalidasi untuk keputusan medis.

Langkah selanjutnya: data eksternal, evaluasi repeated/nested CV, estimasi uncertainty, dan threshold berdasarkan biaya kesalahan yang disepakati bersama ahli domain. Tidak ada klaim bahwa model dengan CV tertinggi unggul secara signifikan dari model lain.

## Repository Structure
```text
predictive-risk-modeling/
├── README.md
├── data/                 # Dokumentasi, snapshot, partisi
├── notebooks/            # EDA, preprocessing, modeling, explainability
├── src/                  # Kode modular yang dapat dijalankan
├── tests/                # Integritas partisi, target, inference dan metrik
├── outputs/
│   ├── figures/
│   ├── tables/
│   ├── executive_summary.md
│   ├── test_metrics.json
│   └── model.joblib
├── requirements.txt
└── requirements-lock.txt
```

## Tools
Python, pandas, NumPy, scikit-learn, matplotlib, joblib, Jupyter.

## Author
Wawan Saputra
