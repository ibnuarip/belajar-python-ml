import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(
    page_title="Student Performance Dashboard",
    page_icon="🎓",
    layout="wide"
)

# Load & bersihkan data
@st.cache_data
def load_data():
    df = pd.read_csv('data/StudentPerformanceFactors.csv')
    df = df.dropna()
    df = df[df['Exam_Score'] <= 100]
    return df

@st.cache_resource
def train_model(df):
    df_model = df.copy()
    le = LabelEncoder()
    kolom_teks = ['Parental_Involvement', 'Access_to_Resources',
                  'Extracurricular_Activities', 'Motivation_Level',
                  'Internet_Access', 'Family_Income', 'Teacher_Quality',
                  'School_Type', 'Peer_Influence', 'Learning_Disabilities',
                  'Parental_Education_Level', 'Distance_from_Home', 'Gender']
    for k in kolom_teks:
        df_model[k] = le.fit_transform(df_model[k])
    X = df_model.drop(columns=['Exam_Score'])
    y = df_model['Exam_Score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, X, r2_score(y_test, y_pred), mean_absolute_error(y_test, y_pred)

df = load_data()
model, X, r2, mae = train_model(df)

# Header
st.title("🎓 Student Performance Dashboard")
st.markdown("Analisis dan prediksi nilai ujian siswa menggunakan Machine Learning")
st.divider()

# Metric Cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Siswa", f"{len(df):,}")
col2.metric("Rata-rata Nilai", f"{df['Exam_Score'].mean():.1f}")
col3.metric("R2 Score", f"{r2:.2f}")
col4.metric("MAE", f"{mae:.2f}")

st.divider()

# Grafik Row 1
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Jam Belajar vs Nilai Ujian")
    fig1 = px.scatter(df, x='Hours_Studied', y='Exam_Score',
                      color='Exam_Score', color_continuous_scale='Blues',
                      opacity=0.5)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("📈 Kehadiran vs Nilai Ujian")
    fig2 = px.scatter(df, x='Attendance', y='Exam_Score',
                      color='Exam_Score', color_continuous_scale='Reds',
                      opacity=0.5)
    st.plotly_chart(fig2, use_container_width=True)

# Grafik Row 2
col3, col4 = st.columns(2)

with col3:
    st.subheader("📉 Distribusi Nilai Ujian")
    fig3 = px.histogram(df, x='Exam_Score', nbins=20,
                        color_discrete_sequence=['steelblue'])
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.subheader("🔥 Korelasi Fitur")
    df_encoded = df.copy()
    le = LabelEncoder()
    for k in df_encoded.select_dtypes(include='object').columns:
        df_encoded[k] = le.fit_transform(df_encoded[k])
    korelasi = df_encoded.corr()['Exam_Score'].drop('Exam_Score').sort_values()
    fig4 = px.bar(x=korelasi.values, y=korelasi.index,
                  orientation='h', color=korelasi.values,
                  color_continuous_scale='RdBu')
    st.plotly_chart(fig4, use_container_width=True)

st.divider()

# Prediksi Interaktif
st.subheader("🔮 Prediksi Nilai Ujian")
st.markdown("Geser slider untuk simulasi prediksi nilai siswa")

col1, col2, col3, col4 = st.columns(4)
with col1:
    jam_belajar = st.slider("Jam Belajar/Minggu", 1, 44, 20)
with col2:
    kehadiran = st.slider("Kehadiran (%)", 60, 100, 80)
with col3:
    nilai_sebelumnya = st.slider("Nilai Sebelumnya", 50, 100, 70)
with col4:
    sesi_les = st.slider("Sesi Les", 0, 10, 2)

input_data = pd.DataFrame([{
    'Hours_Studied': jam_belajar, 'Attendance': kehadiran,
    'Parental_Involvement': 1, 'Access_to_Resources': 1,
    'Extracurricular_Activities': 1, 'Sleep_Hours': 7,
    'Previous_Scores': nilai_sebelumnya, 'Motivation_Level': 1,
    'Internet_Access': 1, 'Tutoring_Sessions': sesi_les,
    'Family_Income': 1, 'Teacher_Quality': 1, 'School_Type': 1,
    'Peer_Influence': 1, 'Physical_Activity': 3,
    'Learning_Disabilities': 0, 'Parental_Education_Level': 1,
    'Distance_from_Home': 1, 'Gender': 1
}])

hasil = model.predict(input_data)[0]
st.metric("🎯 Prediksi Nilai Ujian", f"{hasil:.1f}",
          delta=f"{hasil - df['Exam_Score'].mean():.1f} dari rata-rata")