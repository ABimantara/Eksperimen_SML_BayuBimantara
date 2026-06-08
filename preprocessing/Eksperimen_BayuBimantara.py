# %% [markdown]
# # Eksperimen Machine Learning - Titanic Dataset
# **Nama:** BayuBimantara
# **Dataset:** Titanic Dataset (Kaggle)
# **Problem:** Binary Classification (Survived or Not)

# %% [markdown]
# ## 1. Import Library

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
df = pd.read_csv('../Titanic-Dataset.csv')
print("Shape dataset:", df.shape)
print("\n5 baris pertama:")
df.head()

# %% [markdown]
# ## 3. Exploratory Data Analysis (EDA)

# %%
print("=== INFO DATASET ===")
df.info()

# %%
print("=== STATISTIK DESKRIPTIF ===")
df.describe()

# %%
print("=== CEK MISSING VALUES ===")
print(df.isnull().sum())

# %%
print("=== DISTRIBUSI TARGET (Survived) ===")
print(df['Survived'].value_counts())

# %%
plt.figure(figsize=(6,4))
df['Survived'].value_counts().plot(kind='bar', color=['coral','steelblue'])
plt.title('Distribusi Survived')
plt.xlabel('Survived (0=No, 1=Yes)')
plt.ylabel('Jumlah')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('survived_distribution.png')
plt.show()
print("Plot disimpan!")

# %%
plt.figure(figsize=(8,5))
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.tight_layout()
plt.savefig('missing_values.png')
plt.show()

# %%
plt.figure(figsize=(8,5))
sns.histplot(data=df, x='Age', hue='Survived', bins=30, kde=True)
plt.title('Distribusi Usia berdasarkan Survived')
plt.tight_layout()
plt.savefig('age_distribution.png')
plt.show()

# %%
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Pclass', hue='Survived')
plt.title('Survived berdasarkan Kelas')
plt.tight_layout()
plt.savefig('pclass_survived.png')
plt.show()

# %% [markdown]
# ## 4. Data Preprocessing

# %%
# 4.1 Drop kolom yang tidak relevan
df_clean = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
print("Kolom setelah drop:", df_clean.columns.tolist())

# %%
# 4.2 Handle missing values
df_clean['Age'].fillna(df_clean['Age'].median(), inplace=True)
df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0], inplace=True)
print("Missing values setelah handling:")
print(df_clean.isnull().sum())

# %%
# 4.3 Encoding kolom kategorikal
le = LabelEncoder()
df_clean['Sex'] = le.fit_transform(df_clean['Sex'])
df_clean['Embarked'] = le.fit_transform(df_clean['Embarked'])
print("Encoding selesai!")
df_clean.head()

# %%
# 4.4 Pisahkan fitur dan target
feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = df_clean[feature_cols]
y = df_clean['Survived']

print("Shape X:", X.shape)
print("Shape y:", y.shape)

# %%
# 4.5 Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train:", X_train.shape, "Test:", X_test.shape)

# %%
# 4.6 Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols)

print("Scaling selesai!")
X_train_scaled.head()

# %%
# 4.7 Simpan hasil preprocessing
import os
os.makedirs('../preprocessing', exist_ok=True)

train_preprocessed = X_train_scaled.copy()
train_preprocessed['Survived'] = y_train.values

test_preprocessed = X_test_scaled.copy()
test_preprocessed['Survived'] = y_test.values

train_preprocessed.to_csv('../preprocessing/titanic_preprocessing_train.csv', index=False)
test_preprocessed.to_csv('../preprocessing/titanic_preprocessing_test.csv', index=False)

print("Dataset preprocessing disimpan!")
print("Train:", train_preprocessed.shape)
print("Test:", test_preprocessed.shape)

# %% [markdown]
# ## 5. Quick Model Validation

# %%
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print("=== HASIL QUICK VALIDATION ===")
print(classification_report(y_test, y_pred, target_names=['Not Survived', 'Survived']))

# %%
print("✅ Eksperimen selesai! Dataset sudah siap untuk tahap modelling.")
