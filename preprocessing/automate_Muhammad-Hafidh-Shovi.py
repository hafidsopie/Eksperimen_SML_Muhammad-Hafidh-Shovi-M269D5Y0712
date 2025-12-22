import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler


INPUT_FILE = "stunting_wasting_dataset.csv"

OUTPUT_DIR = "preprocessing"
OUTPUT_FILE = "stunting_wasting_preprocessed.csv"

def preprocess_data(input_path: str):
    print("=== Memulai preprocessing dataset stunting & wasting ===")

    # Load Dataset

    df = pd.read_csv(input_path)


    # Menghapus Duplicate Values

    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    print(f"Data duplikat dihapus: {before - after}")
    
    # Menghapus Fitur yang tidak digunakan
    df = df.drop(['Wasting'], axis=1)

    #  Encoding Jenis Kelamin
    df['Jenis Kelamin'] = df['Jenis Kelamin'].map({
        'Laki-laki': 0,
        'Perempuan': 1
    })
    le_stunting = LabelEncoder()

    df['Stunting'] = le_stunting.fit_transform(df['Stunting'])
    print("Label encoding untuk Stunting dan Wasting selesai.")

# Simpan Hasil Preprocessing

    OUTPUT_FILE = "stunting_wasting_preprocessed.csv"
    df.to_csv(OUTPUT_FILE, index=False)

    print("=== Preprocessing selesai ===")
    print(f"File disimpan di: {OUTPUT_FILE}")

    return df


if __name__ == "__main__":
    preprocess_data(INPUT_FILE)
