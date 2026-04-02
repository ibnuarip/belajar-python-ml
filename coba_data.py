import pandas as pd

# Baca file CSV
df = pd.read_csv('data/StudentPerformanceFactors.csv')

# Lihat 5 baris pertama
print("=== 5 Baris Pertama ===")
print(df.head())

# Lihat info kolom
print("\n=== Info Dataset ===")
print(df.info())

# Lihat statistik dasar
print("\n=== Statistik Dasar ===")
print(df.describe())