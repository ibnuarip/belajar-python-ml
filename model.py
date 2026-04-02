import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Baca dan bersihkan data
df = pd.read_csv('data/StudentPerformanceFactors.csv')
df = df.dropna()
df = df[df['Exam_Score'] <= 100]

# 2. Konversi kolom teks ke angka
le = LabelEncoder()
kolom_teks = ['Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities',
              'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality',
              'School_Type', 'Peer_Influence', 'Learning_Disabilities',
              'Parental_Education_Level', 'Distance_from_Home', 'Gender']

for kolom in kolom_teks:
    df[kolom] = le.fit_transform(df[kolom])

# 3. Cek korelasi terhadap Exam_Score
print("=== Korelasi terhadap Exam_Score ===")
print(df.corr()['Exam_Score'].sort_values(ascending=False))

# 4. Pisahkan input (X) dan target (y)
X = df.drop(columns=['Exam_Score'])
y = df['Exam_Score']

# 5. Split data training & testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Latih model
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Evaluasi
y_pred = model.predict(X_test)
print("\n=== Hasil Evaluasi Model ===")
print(f"R2 Score  : {r2_score(y_test, y_pred):.2f}")
print(f"MAE       : {mean_absolute_error(y_test, y_pred):.2f}")

# 8. Coba prediksi manual
import numpy as np

contoh = pd.DataFrame([{
    'Hours_Studied': 25,
    'Attendance': 90,
    'Parental_Involvement': 2,
    'Access_to_Resources': 2,
    'Extracurricular_Activities': 1,
    'Sleep_Hours': 7,
    'Previous_Scores': 75,
    'Motivation_Level': 2,
    'Internet_Access': 1,
    'Tutoring_Sessions': 2,
    'Family_Income': 1,
    'Teacher_Quality': 2,
    'School_Type': 1,
    'Peer_Influence': 2,
    'Physical_Activity': 3,
    'Learning_Disabilities': 0,
    'Parental_Education_Level': 2,
    'Distance_from_Home': 1,
    'Gender': 1
}])

hasil = model.predict(contoh)
print(f"\n=== Prediksi Manual ===")
print(f"Siswa dengan kehadiran 90% dan belajar 25 jam/minggu")
print(f"Prediksi nilai ujian: {hasil[0]:.1f}")