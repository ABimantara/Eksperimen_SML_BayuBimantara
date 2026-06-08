"""
automate_NamaSiswa.py
Fungsi otomatisasi preprocessing dataset Iris.
Jalankan: python automate_NamaSiswa.py
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os


def load_data() -> pd.DataFrame:
    """Load dataset Iris dan simpan versi raw-nya."""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target

    # Simpan raw
    os.makedirs('..', exist_ok=True)
    df.to_csv('../iris_raw.csv', index=False)
    print("[1/4] Data berhasil di-load. Shape:", df.shape)
    return df, iris.feature_names


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cek dan tangani missing values serta duplikat."""
    # Missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f"  Ditemukan {missing} missing values, mengisi dengan median...")
        df = df.fillna(df.median(numeric_only=True))
    else:
        print("  Tidak ada missing values.")

    # Duplikat
    dup = df.duplicated().sum()
    if dup > 0:
        print(f"  Ditemukan {dup} duplikat, menghapus...")
        df = df.drop_duplicates()
    else:
        print("  Tidak ada duplikat.")

    print("[2/4] Validasi data selesai.")
    return df


def split_data(df: pd.DataFrame, feature_names, test_size: float = 0.2, random_state: int = 42):
    """Pisahkan fitur dan target, lalu split train-test."""
    X = df[list(feature_names)]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[3/4] Split selesai. Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def scale_and_save(X_train, X_test, y_train, y_test, feature_names, output_dir: str = '../preprocessing'):
    """Scaling fitur dan simpan hasil preprocessing."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Konversi ke DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)

    # Gabung dengan target
    train_df = X_train_scaled.copy()
    train_df['target'] = y_train.values

    test_df = X_test_scaled.copy()
    test_df['target'] = y_test.values

    # Simpan
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, 'iris_preprocessing_train.csv')
    test_path = os.path.join(output_dir, 'iris_preprocessing_test.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[4/4] Preprocessing selesai!")
    print(f"  Train disimpan ke: {train_path}")
    print(f"  Test disimpan ke:  {test_path}")
    return train_df, test_df


def run_preprocessing():
    """Fungsi utama yang menjalankan seluruh pipeline preprocessing."""
    print("=" * 50)
    print("  AUTOMATE PREPROCESSING - IRIS DATASET")
    print("=" * 50)

    df, feature_names = load_data()
    df = validate_data(df)
    X_train, X_test, y_train, y_test = split_data(df, feature_names)
    train_df, test_df = scale_and_save(X_train, X_test, y_train, y_test, feature_names)

    print("\n✅ Pipeline preprocessing selesai dijalankan!")
    print(f"   Train: {train_df.shape} | Test: {test_df.shape}")
    return train_df, test_df


if __name__ == "__main__":
    run_preprocessing()
