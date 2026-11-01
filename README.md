# Analisis & Klasterisasi Kota Berdasarkan Indeks Harga Konsumen (IHK) BPS

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Clustering-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-darkblue.svg)](https://pandas.pydata.org/)
[![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)]()

Proyek analisis data science untuk mengelompokkan (*clustering*) kota dan kabupaten di Indonesia berdasarkan karakteristik dan dinamika **Indeks Harga Konsumen (IHK)** dari publikasi data resmi **Badan Pusat Statistik (BPS)** menggunakan algoritma **K-Means Clustering**.

Dikembangkan oleh **Firstyatama Auzan Farhadi** (NIM: 1203220009), S1 Informatika, **Universitas Telkom Surabaya**.

---

## 📊 Metodologi Analisis
1. **Penggabungan Data (`combine.py`)**: Mengonsolidasikan catatan historis harga komoditas dan angka IHK antar-wilayah provinsi ke dalam dataset tabular terpadu (`datanew.csv`).
2. **Prapemrosesan Data**: Penanganan *missing values*, pembersihan format angka desimal lokal, dan standarisasi fitur menggunakan `StandardScaler`.
3. **Penentuan Jumlah Klaster ($K$)**: Evaluasi titik optimal menggunakan metode *Elbow Method* dan *Silhouette Analysis*.
4. **Klasterisasi K-Means**: Pengelompokan daerah ke dalam klaster:
   - Klaster Wilayah IHK Tinggi / Fluktuasi Inflasi Cepat
   - Klaster Wilayah IHK Sedang / Stabil
   - Klaster Wilayah IHK Rendah

---

## 🚀 Panduan Menjalankan

```bash
git clone https://github.com/FirstyatamaAuzan/Prediksi-kota-berdasarkan-IHK.git
cd Prediksi-kota-berdasarkan-IHK
pip install pandas scikit-learn matplotlib seaborn
python combine.py
```

---

## 👤 Pengembang
**Firstyatama Auzan Farhadi**  
- LinkedIn: [linkedin.com/in/firstyatama](https://www.linkedin.com/in/firstyatama/)  
- GitHub: [@FirstyatamaAuzan](https://github.com/FirstyatamaAuzan)  
- S1 Informatika, Universitas Telkom Surabaya
