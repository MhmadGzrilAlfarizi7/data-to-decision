# Studi Kasus: Make Sense of Data with Analysis and AI (PT Maju Bersama)

Berdasarkan kuliah praktisi **Andria Arisal (BRIN, Pusat Riset Sains Data dan Informasi)**, Informatics Practitioner Lecture Series 2026.

**Kasus (dari pemateri):** PT Maju Bersama (distributor barang). Dalam 2 tahun: sales revenue naik, net profit turun,
piutang naik tajam, operating cash flow berkurang. Pimpinan bertanya: *kinerja perusahaan baik atau buruk?*
Tugas analis: mengubah **pertanyaan bisnis → pertanyaan analitik → data → informasi → insight → keputusan**
dengan proses **CRISP-DM** (Business Understanding → Data Understanding → Data Preparation → Modeling → Evaluation → Deployment).

## Struktur
| File | Tahap CRISP-DM | Isi |
|---|---|---|
| `src/00_generate_data.py` | (data) | Membuat database SQLite simulasi (5 tabel, data sengaja "berantakan") |
| `src/01_data_understanding.py` | Data Understanding | Ukuran tabel, granularitas, rentang waktu, masalah kualitas data |
| `src/02_analysis_to_decision.py` | Preparation, Modeling, Evaluation | Informasi (KPI), Insight (diagnostik), Model risiko telat bayar, Keputusan, dashboard |
| `src/common.py` | Data Preparation | Cleaning + penggabungan tabel + perhitungan KPI bulanan |

## Cara menjalankan
```bash
pip install -r requirements.txt
cd src
python 00_generate_data.py
python 01_data_understanding.py
python 02_analysis_to_decision.py
```
Output: teks di terminal, `results/dashboard.png`, `results/kpi_tahun1_vs_tahun2.csv`.

## Hubungan dengan KNIME (tool yang diperkenalkan pemateri)
Pemateri memakai KNIME (workflow visual). Alur di repo ini bisa dipetakan ke node KNIME:
`DB Reader` (baca tabel) → `Duplicate Row Filter` + `Missing Value` + `String Manipulation` (cleaning) → `Joiner` (gabung tabel) →
`GroupBy` (KPI bulanan) → `Partitioning` + `Logistic Regression Learner/Predictor` + `Scorer` (model risiko) → `Line Plot`/`Bar Chart`.

## Data
**Data simulasi**; angka bukan data perusahaan nyata. Setting simulasi: pelanggan distributor besar makin dominan, diskon naik,
harga naik lebih lambat dari HPP, pembayaran makin telat. Karena itu hasilnya mereplikasi pola kasus pemateri
(revenue naik, net profit, kas turun, piutang naik), tetapi nilainya tidak sama dengan slide pemateri.

## Catatan
- Model prediksi (Logistic Regression, ROC-AUC sekitar 0,78) hanya memberi **prioritas penagihan**, bukan keputusan akhir.
  Sesuai pemateri, keputusan butuh informasi eksternal dan pengalaman manusia, bukan hanya data.
- Asumsi arus kas: HPP dibayar ke pemasok 1 bulan setelah pembelian; biaya operasional dibayar pada bulan berjalan.
