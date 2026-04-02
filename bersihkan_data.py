import pandas as pd

# Baca dataset
df = pd.read_csv('data/StudentPerformanceFactors.csv')

# 1. Hapus baris yang ada data kosong
df = df.dropna()

# 2. Hapus data yang tidak masuk akal
df = df[df['Exam_Score'] <= 100]

# 3. Cek hasilnya
print(f"Jumlah data setelah dibersihkan: {len(df)}")
print("\n=== 5 Baris Pertama ===")
print(df.head())
print("\n=== Kolom yang tersisa ===")
print(df.columns.tolist())