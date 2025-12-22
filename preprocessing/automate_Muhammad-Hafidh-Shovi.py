import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler


INPUT_FILE = "stunting_wasting_dataset.csv"
OUTPUT_DIR = "preprocessing"
OUTPUT_FILE = "stunting_wasting_preprocessed.csv"

def preprocess_data(input_path: str):
    print("=== Memulai preprocessing dataset stunting & wasting ===")

    # =====================
    # 1. Load Dataset
    # =====================
    df = pd.read_csv(input_path)

    # =====================
    # 2. Menghapus Duplicate Values
    # =====================
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    print(f"Data duplikat dihapus: {before - after}")

    # =====================
    # 3. Encoding Jenis Kelamin
    # =====================
    df['Jenis Kelamin'] = df['Jenis Kelamin'].map({
        'Laki-laki': 0,
        'Perempuan': 1
    })

    # =====================
    # 4. Encoding Label Stunting & Wasting
    # =====================
    le_stunting = LabelEncoder()
    le_wasting = LabelEncoder()

    df['Stunting'] = le_stunting.fit_transform(df['Stunting'])
    df['Wasting'] = le_wasting.fit_transform(df['Wasting'])
    print("Label encoding untuk Stunting dan Wasting selesai.")
# =====================
# 6. Simpan Hasil Preprocessing
# =====================
    OUTPUT_FILE = "stunting_wasting_preprocessed.csv"
    df.to_csv(OUTPUT_FILE, index=False)

    print("=== Preprocessing selesai ===")
    print(f"File disimpan di: {OUTPUT_FILE}")

    return df


if __name__ == "__main__":
    preprocess_data(INPUT_FILE)

