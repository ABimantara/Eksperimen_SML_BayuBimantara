"""
automate_BayuBimantara.py
Fungsi otomatisasi preprocessing dataset Titanic.
Jalankan: python automate_BayuBimantara.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os


def load_data(filepath: str = '../Titanic-Dataset.csv') -> pd.DataFrame:
    """Load dataset Titanic."""
    df = pd.read_csv(filepath)
    print("[1/4] Data berhasil di-load. Shape:", df.shape)
    return df


def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop kolom tidak relevan dan handle missing values."""
    # Drop kolom tidak relevan
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

    # Handle missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    print("[2/4] Cleaning selesai. Missing values:", df.isnull().sum().sum())
    return df


def encode_and_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Encoding dan split data."""
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    df['Embarked'] = le.fit_transform(df['Embarked'])

    feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    X = df[feature_cols]
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[3/4] Split selesai. Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, feature_cols


def scale_and_save(X_train, X_test, y_train, y_test, feature_cols, output_dir='../preprocessing'):
    """Scaling dan simpan hasil preprocessing."""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)

    train_df = X_train_scaled.copy()
    train_df['Survived'] = y_train.values

    test_df = X_test_scaled.copy()
    test_df['Survived'] = y_test.values

    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(f'{output_dir}/titanic_preprocessing_train.csv', index=False)
    test_df.to_csv(f'{output_dir}/titanic_preprocessing_test.csv', index=False)

    print(f"[4/4] Disimpan ke {output_dir}")
    return train_df, test_df


def run_preprocessing():
    """Pipeline utama preprocessing."""
    print("="*50)
    print("  AUTOMATE PREPROCESSING - TITANIC DATASET")
    print("="*50)

    df = load_data()
    df = validate_and_clean(df)
    X_train, X_test, y_train, y_test, feature_cols = encode_and_split(df)
    train_df, test_df = scale_and_save(X_train, X_test, y_train, y_test, feature_cols)

    print("\n✅ Pipeline preprocessing selesai!")
    print(f"   Train: {train_df.shape} | Test: {test_df.shape}")
    return train_df, test_df


if __name__ == "__main__":
    run_preprocessing()
