# %% [markdown]
# # Eksperimen Machine Learning - Iris Dataset
# **Nama:** NamaSiswa
# **Dataset:** Iris Dataset (sklearn)
# **Problem:** Multi-class Classification

# %% [markdown]
# ## 1. Import Library

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("Library berhasil diimport!")

# %% [markdown]
# ## 2. Data Loading

# %%
# Load dataset Iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("Shape dataset:", df.shape)
print("\n5 baris pertama:")
df.head()

# %%
# Simpan raw dataset
df.to_csv('../iris_raw.csv', index=False)
print("Dataset raw berhasil disimpan ke iris_raw.csv")

# %% [markdown]
# ## 3. Exploratory Data Analysis (EDA)

# %%
# Informasi dasar dataset
print("=== INFO DATASET ===")
print(df.info())

# %%
print("=== STATISTIK DESKRIPTIF ===")
df.describe()

# %%
print("=== CEK MISSING VALUES ===")
print(df.isnull().sum())
print("\nTotal missing values:", df.isnull().sum().sum())

# %%
print("=== DISTRIBUSI KELAS ===")
print(df['species'].value_counts())

# %%
# Visualisasi distribusi kelas
plt.figure(figsize=(6, 4))
df['species'].value_counts().plot(kind='bar', color=['steelblue', 'coral', 'mediumseagreen'])
plt.title('Distribusi Kelas Iris')
plt.xlabel('Spesies')
plt.ylabel('Jumlah')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('class_distribution.png')
plt.show()
print("Plot disimpan!")

# %%
# Visualisasi distribusi fitur
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
features = iris.feature_names

for i, (ax, feat) in enumerate(zip(axes.flatten(), features)):
    for label, color in zip([0, 1, 2], ['steelblue', 'coral', 'mediumseagreen']):
        ax.hist(df[df['target'] == label][feat], alpha=0.6,
                label=iris.target_names[label], color=color, bins=15)
    ax.set_title(feat)
    ax.set_xlabel('Nilai')
    ax.set_ylabel('Frekuensi')
    ax.legend()

plt.suptitle('Distribusi Fitur per Kelas', fontsize=14)
plt.tight_layout()
plt.savefig('feature_distribution.png')
plt.show()
print("Plot distribusi fitur disimpan!")

# %%
# Korelasi antar fitur
plt.figure(figsize=(8, 6))
correlation = df[iris.feature_names].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Heatmap Korelasi Fitur')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()
print("Heatmap korelasi disimpan!")

# %%
# Pairplot
plt.figure(figsize=(10, 8))
sns.pairplot(df[iris.feature_names + ['species']], hue='species',
             palette={'setosa': 'steelblue', 'versicolor': 'coral', 'virginica': 'mediumseagreen'})
plt.suptitle('Pairplot Fitur Iris', y=1.02, fontsize=14)
plt.savefig('pairplot.png', bbox_inches='tight')
plt.show()
print("Pairplot disimpan!")

# %% [markdown]
# ## 4. Data Preprocessing

# %%
# 4.1 Pisahkan fitur dan target
X = df[iris.feature_names]
y = df['target']

print("Shape X:", X.shape)
print("Shape y:", y.shape)

# %%
# 4.2 Cek outlier dengan boxplot
plt.figure(figsize=(12, 5))
for i, feat in enumerate(iris.feature_names):
    plt.subplot(1, 4, i+1)
    plt.boxplot(df[feat])
    plt.title(feat, fontsize=9)
    plt.tight_layout()
plt.suptitle('Boxplot - Deteksi Outlier', fontsize=13, y=1.02)
plt.savefig('boxplot_outlier.png', bbox_inches='tight')
plt.show()
print("Boxplot disimpan!")

# %%
# 4.3 Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Ukuran data training:", X_train.shape)
print("Ukuran data testing:", X_test.shape)

# %%
# 4.4 Feature Scaling dengan StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Konversi kembali ke DataFrame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=iris.feature_names)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=iris.feature_names)

print("Scaling selesai!")
print("\nContoh data setelah scaling:")
X_train_scaled.head()

# %%
# 4.5 Gabungkan kembali dan simpan
train_preprocessed = X_train_scaled.copy()
train_preprocessed['target'] = y_train.values

test_preprocessed = X_test_scaled.copy()
test_preprocessed['target'] = y_test.values

# Simpan ke folder preprocessing
train_preprocessed.to_csv('../preprocessing/iris_preprocessing_train.csv', index=False)
test_preprocessed.to_csv('../preprocessing/iris_preprocessing_test.csv', index=False)

print("Dataset preprocessing disimpan!")
print("Train shape:", train_preprocessed.shape)
print("Test shape:", test_preprocessed.shape)

# %%
# Verifikasi hasil preprocessing
print("=== VERIFIKASI DATA PREPROCESSING ===")
print("\nStatistik data training setelah scaling:")
print(train_preprocessed.describe())

# %% [markdown]
# ## 5. Quick Model Validation (Opsional - hanya untuk cek kualitas preprocessing)

# %%
# Cek apakah data siap dilatih
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print("=== HASIL QUICK VALIDATION ===")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# %%
print("✅ Eksperimen selesai! Dataset sudah siap untuk tahap modelling.")
