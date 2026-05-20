# Preprocessing Ulasan Aplikasi DANA - Analisis Sentimen

Proyek machine learning untuk preprocessing dan analisis sentimen pada dataset ulasan aplikasi DANA dari Google Play Store dengan metode lexicon-based berbahasa Indonesia.

## Deskripsi

Proyek ini menyediakan pipeline preprocessing lengkap untuk:
- Pembersihan teks (lowercase, hapus URL, mention, karakter khusus)
- Tokenisasi menggunakan NLTK
- Penghapusan stopwords bahasa Indonesia
- Stemming menggunakan Sastrawi
- Analisis sentimen berbasis lexicon (positif/negatif/netral)
- Pemisahan data train/validasi/test (70/15/15)
- Export hasil dalam format CSV dan Parquet

## Struktur Folder

```
preprocessing/
├── Eksperimen_Chairul-Ikhsan.ipynb           (Notebook interaktif)
├── automate_Chairul-Ikhsan.py                (Script otomatis)
├── requirements.txt                          (Dependencies)
├── ulasan-aplikasi-dana_preprocessing/       (Output folder)
└── README.md

../
├── ulasan-aplikasi-dana_raw/
│   └── ulasan-aplikasi-dana.csv              (Dataset raw)
└── README.md                                 (Root dokumentasi)
```

## Requirements

```
Python 3.12+
pandas
numpy
matplotlib
seaborn
wordcloud
nltk
scikit-learn
sastrawi
```

## Instalasi

1. Navigasi ke folder proyek:
```bash
cd preprocessing
```

2. Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn wordcloud nltk scikit-learn sastrawi
```

3. Pastikan dataset tersedia:
```bash
../ulasan-aplikasi-dana_raw/ulasan-aplikasi-dana.csv
```

## Cara Menjalankan

### Opsi 1: Menggunakan Notebook (Interaktif)

```bash
jupyter notebook Eksperimen_Chairul-Ikhsan.ipynb
```

Jalankan semua cell secara berurutan atau per-section untuk melihat visualisasi dan analisis detail.

### Opsi 2: Menggunakan Script Automation (Non-interaktif)

```bash
python automate_Chairul-Ikhsan.py
```

Script akan menjalankan seluruh pipeline preprocessing dan menampilkan summary hasil akhir.

## Output

Setelah preprocessing selesai, folder `ulasan-aplikasi-dana_preprocessing/` akan berisi:

- `data_preprocessed.csv/` - Dataset lengkap setelah preprocessing
- `X_train.csv/` - Teks training
- `X_val.csv/` - Teks validasi
- `X_test.csv/` - Teks testing
- `y_train.csv/` - Label training
- `y_val.csv/` - Label validasi
- `y_test.csv/` - Label testing

## Tahap Preprocessing

1. **Memuat dataset** - Membaca file CSV dengan handling untuk bad lines
2. **Menangani nilai kosong** - Menghapus baris dengan kolom 'content' kosong
3. **Menangani duplikat** - Menghapus duplikat berdasarkan kolom 'content'
4. **Pembersihan teks** - Huruf kecil, hapus URL, mentions, hashtag, karakter khusus
5. **Tokenisasi** - Memisahkan teks menjadi token kata menggunakan NLTK
6. **Penghapusan stopwords** - Menghapus stopwords bahasa Indonesia, panjang token > 1
7. **Stemming** - Menerapkan stemming dengan Sastrawi untuk normalisasi kata
8. **Penggabungan token** - Menggabungkan token yang telah diproses menjadi string
9. **Analisis sentimen** - Pelabelan sentimen berbasis lexicon (positive/negative/neutral)
10. **Pemisahan data** - Split train/validasi/test (70/15/15) dengan stratifikasi
11. **Export hasil** - Menyimpan dalam format CSV

