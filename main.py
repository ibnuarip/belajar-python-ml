"""
Beginner-friendly Machine Learning example with scikit-learn.

This script shows a full ML workflow:
1) Load data
2) Split data into training and testing sets
3) Train a model
4) Evaluate model performance
"""
# Contoh Machine Learning yang ramah pemula menggunakan scikit-learn.
# Skrip ini menunjukkan alur kerja ML lengkap dari awal sampai evaluasi.

# Import dataset loader for a simple real-world dataset (Iris flowers).
from sklearn.datasets import load_iris

# Import helper to split data into train/test sets.
from sklearn.model_selection import train_test_split

# Import a simple classification model.
from sklearn.linear_model import LogisticRegression

# Import evaluation metrics to measure model quality.
from sklearn.metrics import accuracy_score, classification_report


# ---------------------------
# 1) LOAD THE DATASET
# ---------------------------
# load_iris() returns features (X), labels (y), and metadata.
iris = load_iris()

# X contains numeric inputs (features) like petal length, sepal width, etc.
X = iris.data

# y contains target labels (0, 1, 2) for flower species.
y = iris.target

# target_names maps numeric labels to human-readable names.
target_names = iris.target_names


# ---------------------------
# 2) SPLIT DATA (TRAIN/TEST)
# ---------------------------
# We keep 20% for testing, 80% for training.
# random_state makes result reproducible (same split every run).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ---------------------------
# 3) TRAIN THE MODEL
# ---------------------------
# LogisticRegression is a simple and common classifier.
# max_iter is increased to ensure convergence on some environments.
model = LogisticRegression(max_iter=200)

# Learn patterns from training data.
model.fit(X_train, y_train)


# ---------------------------
# 4) MAKE PREDICTIONS
# ---------------------------
# Predict labels for unseen test data.
y_pred = model.predict(X_test)


# ---------------------------
# 5) EVALUATE THE MODEL
# ---------------------------
# Accuracy = percentage of correct predictions.
accuracy = accuracy_score(y_test, y_pred)


# ---------------------------
# 6) PRINT RESULTS (READABLE)
# ---------------------------
print("=" * 50)
# Menampilkan judul utama agar output mudah dibaca.
print("MACHINE LEARNING WORKFLOW (scikit-learn)")
# Menampilkan garis pembatas atas dan bawah judul.
print("=" * 50)
# Menampilkan nama dataset yang digunakan.
print(f"Dataset           : Iris")
# Menampilkan jumlah total data pada dataset.
print(f"Total samples     : {len(X)}")
# Menampilkan jumlah data latih (training set).
print(f"Training samples  : {len(X_train)}")
# Menampilkan jumlah data uji (testing set).
print(f"Testing samples   : {len(X_test)}")
# Menampilkan model yang dipakai untuk klasifikasi.
print(f"Model             : LogisticRegression")
# Menampilkan garis pemisah antar bagian output.
print("-" * 50)
# Menampilkan nilai akurasi dalam format persen.
print(f"Accuracy          : {accuracy:.2%}")
# Menampilkan garis pemisah sebelum laporan detail.
print("-" * 50)
# Menampilkan judul untuk laporan klasifikasi detail.
print("Detailed Classification Report:")
# Mencetak laporan klasifikasi yang berisi precision, recall, f1-score, dan support.
print(
    # Fungsi ini menghitung metrik evaluasi untuk setiap kelas target.
    classification_report(
        # Label aktual dari data uji.
        y_test,
        # Label prediksi hasil model.
        y_pred,
        # Nama kelas agar output lebih mudah dipahami.
        target_names=target_names,
    )
)

# Bonus: show a few example predictions in a readable way.
print("-" * 50)
# Menampilkan judul bagian contoh prediksi.
print("Sample Predictions (first 5 test rows):")
# Melakukan perulangan untuk menampilkan 5 contoh prediksi pertama.
for i in range(5):
    # Mengambil nama kelas aktual berdasarkan label asli.
    actual_name = target_names[y_test[i]]
    # Mengambil nama kelas prediksi berdasarkan hasil model.
    predicted_name = target_names[y_pred[i]]
    # Menampilkan perbandingan kelas aktual vs prediksi per baris.
    print(f"Row {i + 1}: Actual = {actual_name:<10} | Predicted = {predicted_name}")
