import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import csv
import json
from datetime import datetime
import requests
from io import StringIO
from pathlib import Path
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

plt.style.use('default')
sns.set_palette('husl')


# Fungsi Pembersihan dan Preprocessing Teks
def bersihkan_teks(text):
    """
    Bersihkan teks dengan menerapkan beberapa langkah preprocessing:
    - Ubah ke huruf kecil
    - Hapus URL
    - Hapus mention dan hashtag
    - Hapus karakter non-alfabet (hanya a-z dan spasi)
    - Hapus spasi berlebih

    Args:
        text (str): Teks mentah untuk dibersihkan

    Returns:
        str: Teks yang telah dibersihkan
    """
    if pd.isna(text):
        return ''

    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def tokenize_text(text):
    """
    Tokenisasi teks menjadi daftar kata.

    Args:
        text (str): Teks yang akan ditokenisasi

    Returns:
        list: Daftar token hasil tokenisasi
    """
    return word_tokenize(text) if text else []


def remove_stopwords(tokens):
    """
    Hapus stopwords dari daftar token dan filter token dengan panjang lebih dari 1.

    Args:
        tokens (list): Daftar token

    Returns:
        list: Token yang telah difilter
    """
    stop_words = set(stopwords.words('indonesian'))
    return [token for token in tokens if token not in stop_words and len(token) > 1]


def stem_tokens(tokens, stemmer):
    """
    Terapkan stemming pada daftar token menggunakan stemmer Sastrawi Indonesia.

    Args:
        tokens (list): Daftar token
        stemmer: Objek stemmer Sastrawi yang sudah dibuat

    Returns:
        list: Token yang telah di-stem
    """
    return [stemmer.stem(token) for token in tokens]


# Fungsi Pelabelan Sentimen dengan Lexicon
def load_lexicons():
    """
    Memuat kamus kata positif dan negatif dari file lokal dan GitHub.

    Returns:
        tuple: (lexicon_positive dict, lexicon_negative dict)
    """
    lexicon_positive = dict()
    lexicon_negative = dict()

    # Load dari file lokal
    local_path = Path('lexicon/lexicon_dana.csv')
    if local_path.exists():
        df_lexicon = pd.read_csv(local_path)
        for _, row in df_lexicon.iterrows():
            if row['sentimen'] == 'positif':
                lexicon_positive[row['kata']] = 1
            elif row['sentimen'] == 'negatif':
                lexicon_negative[row['kata']] = -1
        print(f'Lexicon lokal - positif: {len(lexicon_positive)}, negatif: {len(lexicon_negative)}')
    else:
        print('File lokal tidak ditemukan')

    # Download dari GitHub
    response = requests.get('https://raw.githubusercontent.com/angelmetanosaa/dataset/main/lexicon_positive.csv')
    if response.status_code == 200:
        reader = csv.reader(StringIO(response.text), delimiter=',')
        for row in reader:
            lexicon_positive[row[0]] = int(row[1])
    else:
        print('Gagal mengunduh kamus kata positif dari GitHub')

    response = requests.get('https://raw.githubusercontent.com/angelmetanosaa/dataset/main/lexicon_negative.csv')
    if response.status_code == 200:
        reader = csv.reader(StringIO(response.text), delimiter=',')
        for row in reader:
            lexicon_negative[row[0]] = int(row[1])
    else:
        print('Gagal mengunduh kamus kata negatif dari GitHub')

    print(f'Jumlah kata dalam kamus positif : {len(lexicon_positive)}')
    print(f'Jumlah kata dalam kamus negatif : {len(lexicon_negative)}')

    return lexicon_positive, lexicon_negative


def sentiment_analysis_lexicon_indonesia(text, lexicon_positive, lexicon_negative):
    """
    Lakukan analisis sentimen menggunakan pendekatan berbasis lexicon.
    Setiap token dalam teks dicocokkan dengan kamus positif dan negatif.
    Skor akhir menentukan polaritas sentimen.

    Args:
        text (list or str): Teks atau daftar token yang akan dianalisis
        lexicon_positive (dict): Kamus kata positif beserta skornya
        lexicon_negative (dict): Kamus kata negatif beserta skornya

    Returns:
        tuple: (sentiment_score, polarity)
    """
    # Mendukung input berupa list token maupun string
    if isinstance(text, str):
        tokens = text.split()
    else:
        tokens = text

    score = 0

    # Menjumlahkan skor kata-kata yang ada di kamus positif
    for word in tokens:
        if word in lexicon_positive:
            score = score + lexicon_positive[word]

    # Menjumlahkan skor kata-kata yang ada di kamus negatif
    for word in tokens:
        if word in lexicon_negative:
            score = score + lexicon_negative[word]

    # Menentukan polaritas berdasarkan nilai skor akhir
    if score > 0:
        polarity = 'positive'
    elif score < 0:
        polarity = 'negative'
    else:
        polarity = 'neutral'

    return score, polarity


# Fungsi Visualisasi EDA
def plot_distribusi_skor(df, output_dir=None):
    """
    Membuat visualisasi distribusi skor penilaian pengguna.

    Args:
        df (DataFrame): Dataset dengan kolom 'score'
        output_dir (Path, optional): Direktori untuk menyimpan gambar
    """
    score_counts = df['score'].value_counts().sort_index()

    plt.figure(figsize=(10, 6))
    score_counts.plot(kind='bar', color='steelblue', edgecolor='black')
    plt.title('Distribusi Skor Ulasan', fontsize=14, fontweight='bold')
    plt.xlabel('Skor', fontsize=12)
    plt.ylabel('Jumlah', fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if output_dir:
        plt.savefig(output_dir / 'distribusi_skor.png', dpi=150)
    plt.show()
    plt.close()


def plot_panjang_ulasan(df, output_dir=None):
    """
    Membuat visualisasi distribusi panjang karakter ulasan.

    Args:
        df (DataFrame): Dataset dengan kolom 'review_length'
        output_dir (Path, optional): Direktori untuk menyimpan gambar
    """
    plt.figure(figsize=(10, 6))
    plt.hist(df['review_length'], bins=50, color='coral', edgecolor='black', alpha=0.7)
    plt.title('Distribusi Panjang Ulasan', fontsize=14, fontweight='bold')
    plt.xlabel('Jumlah Karakter', fontsize=12)
    plt.ylabel('Frekuensi', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if output_dir:
        plt.savefig(output_dir / 'distribusi_panjang_ulasan.png', dpi=150)
    plt.show()
    plt.close()


def plot_distribusi_sentimen(df_clean, output_dir=None):
    """
    Membuat visualisasi distribusi label sentimen berupa bar chart dan pie chart.

    Args:
        df_clean (DataFrame): Dataset dengan kolom 'polarity'
        output_dir (Path, optional): Direktori untuk menyimpan gambar
    """
    sentiment_counts = df_clean['polarity'].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sentiment_counts.plot(
        kind='bar', ax=axes[0],
        color=['green', 'gray', 'red'],
        edgecolor='black'
    )
    axes[0].set_title('Distribusi Sentimen (Grafik Batang)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Sentimen', fontsize=11)
    axes[0].set_ylabel('Jumlah', fontsize=11)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', alpha=0.3)

    colors = ['green', 'gray', 'red']
    axes[1].pie(
        sentiment_counts.values,
        labels=sentiment_counts.index,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 11}
    )
    axes[1].set_title('Distribusi Sentimen (Grafik Pie)', fontsize=12, fontweight='bold')

    plt.tight_layout()

    if output_dir:
        plt.savefig(output_dir / 'distribusi_sentimen.png', dpi=150)
    plt.show()
    plt.close()


def plot_wordcloud(df_clean, output_dir=None):
    """
    Membuat word cloud untuk ulasan positif dan negatif secara terpisah.

    Args:
        df_clean (DataFrame): Dataset dengan kolom 'text_akhir' dan 'polarity'
        output_dir (Path, optional): Direktori untuk menyimpan gambar
    """
    from wordcloud import WordCloud

    positive_text = ' '.join(
        df_clean[df_clean['polarity'] == 'positive']['text_akhir'].tolist()
    )
    negative_text = ' '.join(
        df_clean[df_clean['polarity'] == 'negative']['text_akhir'].tolist()
    )

    if positive_text.strip():
        wc_pos = WordCloud(
            width=1000, height=500,
            background_color='white',
            colormap='Greens',
            max_words=80
        ).generate(positive_text)

        plt.figure(figsize=(14, 6))
        plt.imshow(wc_pos, interpolation='bilinear')
        plt.axis('off')
        plt.title('WordCloud Ulasan Positif', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()

        if output_dir:
            plt.savefig(output_dir / 'wordcloud_positif.png', dpi=150)
        plt.show()
        plt.close()

    if negative_text.strip():
        wc_neg = WordCloud(
            width=1000, height=500,
            background_color='white',
            colormap='Reds',
            max_words=80
        ).generate(negative_text)

        plt.figure(figsize=(14, 6))
        plt.imshow(wc_neg, interpolation='bilinear')
        plt.axis('off')
        plt.title('WordCloud Ulasan Negatif', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()

        if output_dir:
            plt.savefig(output_dir / 'wordcloud_negatif.png', dpi=150)
        plt.show()
        plt.close()


# Pipeline Preprocessing Utama
def preprocess_dataset(input_path, output_dir=None, simpan_visualisasi=True):
    """
    Pipeline preprocessing lengkap untuk dataset ulasan aplikasi DANA.

    Melakukan tahapan berikut secara berurutan:
    1. Memuat file CSV
    2. Menangani nilai kosong
    3. Menangani duplikat
    4. Membersihkan teks
    5. Tokenisasi
    6. Menghapus stopwords
    7. Stemming menggunakan Sastrawi
    8. Menggabungkan kembali token menjadi string
    9. Pelabelan sentimen berbasis lexicon dari GitHub
    10. Membagi data menjadi train, validasi, dan test (70/15/15)
    11. Menyimpan hasil ke file CSV

    Args:
        input_path (str or Path): Path ke file CSV dataset
        output_dir (str or Path, optional): Direktori untuk menyimpan hasil.
            Default: '../preprocessing/ulasan-aplikasi-dana_preprocessing'
        simpan_visualisasi (bool): Apakah menyimpan plot ke file. Default: True

    Returns:
        dict: Dictionary berisi split data dan statistik preprocessing
    """
    input_path = Path(input_path)

    if output_dir is None:
        output_dir = Path('..') / 'preprocessing' / 'ulasan-aplikasi-dana_preprocessing'
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    vis_dir = output_dir / 'visualisasi' if simpan_visualisasi else None
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)

    # Langkah 1: Memuat dataset
    print('Langkah 1: Memuat dataset')
    df = pd.read_csv(input_path, on_bad_lines='skip')
    print(f'  Jumlah baris   : {df.shape[0]}')
    print(f'  Jumlah kolom   : {df.shape[1]}')
    print(f'  Nama kolom     : {list(df.columns)}')

    # EDA: Info, statistik, nilai kosong, distribusi skor
    print('\nEDA: Info Dataset')
    print(df.info())

    print('\nEDA: Statistik Deskriptif')
    print(df.describe())

    print('\nEDA: Nilai Kosong per Kolom')
    missing = df.isnull().sum()
    print(missing[missing > 0])
    print(f'Total nilai kosong: {missing.sum()}')

    print('\nEDA: Distribusi Skor')
    print(df['score'].value_counts().sort_index())

    df['review_length'] = df['content'].fillna('').str.len()
    print(f'\nRata-rata panjang ulasan : {df["review_length"].mean():.2f} karakter')
    print(f'Panjang ulasan minimum   : {df["review_length"].min()} karakter')
    print(f'Panjang ulasan maksimum  : {df["review_length"].max()} karakter')

    plot_distribusi_skor(df, vis_dir)
    plot_panjang_ulasan(df, vis_dir)

    print('\nEDA: Contoh 5 Ulasan Pertama')
    for idx in range(min(5, len(df))):
        print(f'Ulasan {idx + 1}:')
        print(f'  Skor : {df.iloc[idx]["score"]}')
        print(f'  Isi  : {df.iloc[idx]["content"]}')
        print('-' * 80)

    # Langkah 2: Tangani nilai kosong
    print('\nLangkah 2: Tangani nilai kosong')
    df_clean = df.dropna(subset=['content']).copy()
    df_clean = df_clean[df_clean['content'].str.strip() != ''].copy()
    print(f'  Baris sebelum : {df.shape[0]}')
    print(f'  Baris sesudah : {df_clean.shape[0]}')
    print(f'  Baris dihapus : {df.shape[0] - df_clean.shape[0]}')

    # Langkah 3: Tangani duplikat
    print('\nLangkah 3: Tangani duplikat')
    sebelum_dedup = df_clean.shape[0]
    df_clean = df_clean.drop_duplicates(
        subset=['content'], keep='first'
    ).reset_index(drop=True)
    print(f'  Baris sebelum : {sebelum_dedup}')
    print(f'  Baris sesudah : {df_clean.shape[0]}')
    print(f'  Duplikat dihapus : {sebelum_dedup - df_clean.shape[0]}')

    # Langkah 4: Pembersihan teks
    print('\nLangkah 4: Pembersihan teks')
    df_clean['text_cleaned'] = df_clean['content'].apply(bersihkan_teks)

    print('Sampel pembersihan teks (3 pertama):')
    for idx in range(min(3, len(df_clean))):
        print(f'  Asli   : {df_clean.iloc[idx]["content"]}')
        print(f'  Bersih : {df_clean.iloc[idx]["text_cleaned"]}')
        print('  ' + '-' * 60)

    # Langkah 5: Tokenisasi
    print('\nLangkah 5: Tokenisasi')
    df_clean['tokens'] = df_clean['text_cleaned'].apply(tokenize_text)

    print('Sampel tokenisasi (3 pertama):')
    for idx in range(min(3, len(df_clean))):
        print(f'  Teks bersih : {df_clean.iloc[idx]["text_cleaned"]}')
        print(f'  Token       : {df_clean.iloc[idx]["tokens"]}')
        print('  ' + '-' * 60)

    # Langkah 6: Hapus stopwords
    print('\nLangkah 6: Hapus stopwords')
    df_clean['tokens_no_stopwords'] = df_clean['tokens'].apply(remove_stopwords)

    print('Sampel penghapusan stopwords (3 pertama):')
    for idx in range(min(3, len(df_clean))):
        print(f'  Dengan stopwords : {df_clean.iloc[idx]["tokens"]}')
        print(f'  Tanpa stopwords  : {df_clean.iloc[idx]["tokens_no_stopwords"]}')
        print('  ' + '-' * 60)

    # Langkah 7: Stemming
    print('\nLangkah 7: Stemming')
    stemmer = StemmerFactory().create_stemmer()
    df_clean['tokens_stemmed'] = df_clean['tokens_no_stopwords'].apply(
        lambda tokens: stem_tokens(tokens, stemmer)
    )

    print('Sampel stemming (3 pertama):')
    for idx in range(min(3, len(df_clean))):
        print(f'  Sebelum stemming : {df_clean.iloc[idx]["tokens_no_stopwords"]}')
        print(f'  Sesudah stemming : {df_clean.iloc[idx]["tokens_stemmed"]}')
        print('  ' + '-' * 60)

    # Langkah 8: Gabungkan token menjadi teks akhir
    print('\nLangkah 8: Gabungkan token')
    df_clean['text_akhir'] = df_clean['tokens_stemmed'].apply(
        lambda tokens: ' '.join(tokens)
    )

    print('Sampel teks akhir (3 pertama):')
    for idx in range(min(3, len(df_clean))):
        print(f'  Token     : {df_clean.iloc[idx]["tokens_stemmed"]}')
        print(f'  Teks akhir: {df_clean.iloc[idx]["text_akhir"]}')
        print('  ' + '-' * 60)

    # Langkah 9: Pelabelan sentimen berbasis lexicon
    print('\nLangkah 9: Pelabelan sentimen berbasis lexicon')
    lexicon_positive, lexicon_negative = load_lexicons_from_github()

    df_clean[['polarity_score', 'polarity']] = df_clean['tokens_no_stopwords'].apply(
        lambda tokens: pd.Series(
            sentiment_analysis_lexicon_indonesia(tokens, lexicon_positive, lexicon_negative)
        )
    )

    print('Distribusi label sentimen:')
    print(df_clean['polarity'].value_counts())
    print('\nPersentase:')
    print((df_clean['polarity'].value_counts(normalize=True) * 100).round(2))

    plot_distribusi_sentimen(df_clean, vis_dir)
    plot_wordcloud(df_clean, vis_dir)

    # Langkah 10: Membagi data train, validasi, test (70/15/15)
    print('\nLangkah 10: Membagi data (70% train, 15% validasi, 15% test)')
    X = df_clean['text_akhir'].values
    y = df_clean['polarity'].values

    print(f'Total sampel          : {len(X)}')
    print(f'Distribusi kelas      : {pd.Series(y).value_counts().to_dict()}')

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    print(f'  Train      : {len(X_train)} ({len(X_train) / len(X) * 100:.1f}%)')
    print(f'  Validasi   : {len(X_val)} ({len(X_val) / len(X) * 100:.1f}%)')
    print(f'  Test       : {len(X_test)} ({len(X_test) / len(X) * 100:.1f}%)')

    print('\nVerifikasi distribusi kelas:')
    for nama, label_array in [('Train', y_train), ('Validasi', y_val), ('Test', y_test)]:
        dist = pd.Series(label_array).value_counts()
        pct = (dist / len(label_array) * 100).round(2)
        print(f'  {nama}:')
        for sentimen in dist.index:
            print(f'    {sentimen}: {dist[sentimen]} ({pct[sentimen]}%)')

    # Langkah 11: Menyimpan hasil preprocessing
    print(f'\nLangkah 11: Menyimpan hasil ke {output_dir}')

    preprocessed_data = df_clean[[
        'content', 'score', 'text_akhir', 'polarity', 'polarity_score'
    ]].copy()
    preprocessed_data.to_csv(output_dir / 'data_preprocessed.csv', index=False)
    print(f'  Tersimpan: data_preprocessed.csv ({preprocessed_data.shape[0]} baris)')

    pd.DataFrame(X_train, columns=['text']).to_csv(output_dir / 'X_train.csv', index=False)
    pd.DataFrame(X_val, columns=['text']).to_csv(output_dir / 'X_val.csv', index=False)
    pd.DataFrame(X_test, columns=['text']).to_csv(output_dir / 'X_test.csv', index=False)

    pd.DataFrame(y_train, columns=['polarity']).to_csv(output_dir / 'y_train.csv', index=False)
    pd.DataFrame(y_val, columns=['polarity']).to_csv(output_dir / 'y_val.csv', index=False)
    pd.DataFrame(y_test, columns=['polarity']).to_csv(output_dir / 'y_test.csv', index=False)

    print('  Tersimpan: X_train.csv, X_val.csv, X_test.csv')
    print('  Tersimpan: y_train.csv, y_val.csv, y_test.csv')

    # Langkah 12: Membuat metadata.json
    print('\nLangkah 12: Membuat metadata.json')
    sentiment_dist = df_clean['polarity'].value_counts().to_dict()
    
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'preprocessing_info': {
            'original_rows': int(df.shape[0]),
            'original_columns': int(df.shape[1]),
            'final_rows': int(df_clean.shape[0]),
            'final_columns': int(df_clean.shape[1]),
            'rows_retained_percentage': float((df_clean.shape[0] / df.shape[0] * 100))
        },
        'data_split': {
            'train_samples': int(len(X_train)),
            'train_percentage': float((len(X_train) / len(X) * 100)),
            'validation_samples': int(len(X_val)),
            'validation_percentage': float((len(X_val) / len(X) * 100)),
            'test_samples': int(len(X_test)),
            'test_percentage': float((len(X_test) / len(X) * 100)),
            'random_state': 42,
            'stratify': True
        },
        'sentiment_distribution': {
            'positive': int(sentiment_dist.get('positive', 0)),
            'negative': int(sentiment_dist.get('negative', 0)),
            'neutral': int(sentiment_dist.get('neutral', 0))
        },
        'preprocessing_steps': [
            'Tangani nilai kosong (hapus isi kosong)',
            'Tangani duplikat (hapus isi duplikat)',
            'Pembersihan teks (huruf kecil, hapus URL, mentions, karakter khusus)',
            'Tokenisasi (pisahkan menjadi kata)',
            'Penghapusan stopwords (stopwords Indonesia, panjang > 1)',
            'Stemming (stemmer Sastrawi)',
            'Penggabungan token (buat teks akhir)',
            'Pelabelan sentimen (lexicon-based)',
            'Pemisahan data (70% train, 15% val, 15% test dengan stratifikasi)'
        ],
        'output_files': [
            'data_preprocessed.csv',
            'X_train.csv',
            'X_val.csv',
            'X_test.csv',
            'y_train.csv',
            'y_val.csv',
            'y_test.csv',
            'metadata.json'
        ]
    }
    
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f'  Tersimpan: metadata.json')

    results = {
        'df_clean': df_clean,
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'output_dir': output_dir,
        'original_shape': df.shape,
        'final_shape': df_clean.shape,
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'metadata': metadata
    }

    return results


# Fungsi Ringkasan Hasil
def print_preprocessing_summary(results):
    """
    Menampilkan ringkasan lengkap hasil preprocessing.

    Args:
        results (dict): Dictionary yang dikembalikan dari fungsi preprocess_dataset
    """
    print('=' * 80)
    print('RINGKASAN PREPROCESSING')
    print('=' * 80)

    print('\nPerbandingan Bentuk Data:')
    print(f'  Dataset asli           : {results["original_shape"][0]} baris, {results["original_shape"][1]} kolom')
    print(f'  Setelah preprocessing  : {results["final_shape"][0]} baris, {results["final_shape"][1]} kolom')
    pct_retained = results["final_shape"][0] / results["original_shape"][0] * 100
    print(f'  Baris yang dipertahankan: {results["final_shape"][0]} ({pct_retained:.1f}%)')

    print('\nTahap Preprocessing yang Diterapkan:')
    stages = [
        '1. Tangani nilai kosong (hapus isi kosong)',
        '2. Tangani duplikat (hapus isi duplikat)',
        '3. Pembersihan teks (huruf kecil, hapus URL, mentions, karakter khusus)',
        '4. Tokenisasi (pisahkan menjadi kata)',
        '5. Penghapusan stopwords (stopwords Indonesia, panjang > 1)',
        '6. Stemming (stemmer Sastrawi)',
        '7. Penggabungan token (buat teks akhir)',
        '8. Pelabelan sentimen (lexicon-based dari GitHub)',
        '9. Pemisahan data (70% train, 15% val, 15% test dengan stratifikasi)',
    ]
    for stage in stages:
        print(f'  {stage}')

    print('\nFile Output yang Dibuat:')
    output_files = [
        'data_preprocessed.csv',
        'X_train.csv, X_val.csv, X_test.csv',
        'y_train.csv, y_val.csv, y_test.csv',
    ]
    for f in output_files:
        print(f'  {f}')

    print(f'\nDirektori output: {results["output_dir"]}')

    total = results['train_size'] + results['val_size'] + results['test_size']
    print('\nStatistik Pembagian Data:')
    print(f'  Train    : {results["train_size"]} sampel ({results["train_size"] / total * 100:.1f}%)')
    print(f'  Validasi : {results["val_size"]} sampel ({results["val_size"] / total * 100:.1f}%)')
    print(f'  Test     : {results["test_size"]} sampel ({results["test_size"] / total * 100:.1f}%)')

    print('\nDistribusi Sentimen (Dataset Akhir):')
    df_clean = results['df_clean']
    for sentimen in ['positive', 'negative', 'neutral']:
        count = (df_clean['polarity'] == sentimen).sum()
        pct = (count / len(df_clean) * 100) if len(df_clean) > 0 else 0
        print(f'  {sentimen}: {count} ({pct:.1f}%)')

    print('\n' + '=' * 80)
    print('Preprocessing selesai.')


# Entry Point
if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        input_file = '../ulasan-aplikasi-dana_raw/ulasan-aplikasi-dana.csv'
        print(f'Menggunakan path input default: {input_file}')
    else:
        input_file = sys.argv[1]

    if len(sys.argv) < 3:
        output_directory = None
        print('Menggunakan direktori output default: ../preprocessing/ulasan-aplikasi-dana_preprocessing')
    else:
        output_directory = sys.argv[2]

    results = preprocess_dataset(input_file, output_directory, simpan_visualisasi=True)
    print_preprocessing_summary(results)