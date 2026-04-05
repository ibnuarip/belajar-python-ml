import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np

# ── Page Config ─────────────────────────────────────
st.set_page_config(
    page_title="EduMetrics — Student Performance",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

* { font-family: 'DM Sans', sans-serif; }

/* Background */
.stApp {
    background: #0a0e1a;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f1629 !important;
    border-right: 1px solid rgba(255,255,255,0.06);
}

/* Metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #141d35 0%, #0f1629 100%);
    border: 1px solid rgba(99, 179, 237, 0.15);
    border-radius: 16px;
    padding: 20px 24px;
}

[data-testid="stMetricLabel"] {
    color: #7a8fa6 !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

[data-testid="stMetricValue"] {
    color: #e8f4fd !important;
    font-size: 32px !important;
    font-weight: 600 !important;
    font-family: 'DM Mono', monospace !important;
}

[data-testid="stMetricDelta"] {
    font-size: 13px !important;
}

/* Headers */
h1 { color: #e8f4fd !important; font-weight: 600 !important; letter-spacing: -0.02em !important; }
h2, h3 { color: #c5d8ed !important; font-weight: 500 !important; }

/* Slider */
.stSlider > div > div > div {
    background: #1e3a5f !important;
}

/* Section card */
.section-card {
    background: linear-gradient(135deg, #141d35 0%, #0f1629 100%);
    border: 1px solid rgba(99, 179, 237, 0.1);
    border-radius: 20px;
    padding: 28px 32px;
    margin-bottom: 24px;
}

.section-title {
    color: #7a8fa6;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 4px;
}

.section-heading {
    color: #e8f4fd;
    font-size: 22px;
    font-weight: 600;
    letter-spacing: -0.02em;
    margin-bottom: 20px;
}

.prediction-result {
    background: linear-gradient(135deg, #1a3a5c 0%, #0f2240 100%);
    border: 1px solid rgba(99, 179, 237, 0.3);
    border-radius: 20px;
    padding: 32px;
    text-align: center;
    margin-top: 24px;
}

.prediction-score {
    font-family: 'DM Mono', monospace;
    font-size: 72px;
    font-weight: 600;
    color: #63b3ed;
    line-height: 1;
    margin: 8px 0;
}

.prediction-label {
    color: #7a8fa6;
    font-size: 13px;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 500;
    margin-top: 12px;
}

.badge-good { background: rgba(72, 187, 120, 0.15); color: #68d391; border: 1px solid rgba(72,187,120,0.3); }
.badge-avg  { background: rgba(236, 201, 75, 0.15);  color: #f6e05e; border: 1px solid rgba(236,201,75,0.3); }
.badge-low  { background: rgba(252, 129, 74, 0.15);  color: #fc8181; border: 1px solid rgba(252,129,74,0.3); }

/* Divider */
hr { border-color: rgba(255,255,255,0.06) !important; }

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load & Process Data ──────────────────────────────
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
    return model, X, r2_score(y_test, y_pred), mean_absolute_error(y_test, y_pred), X_train, X_test, y_train, y_test

df = load_data()
model, X, r2, mae, X_train, X_test, y_train, y_test = train_model(df)

# Plotly theme
PLOTLY_THEME = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='DM Sans', color='#7a8fa6', size=12),
    xaxis=dict(gridcolor='rgba(255,255,255,0.05)', linecolor='rgba(255,255,255,0.08)', tickfont=dict(color='#7a8fa6')),
    yaxis=dict(gridcolor='rgba(255,255,255,0.05)', linecolor='rgba(255,255,255,0.08)', tickfont=dict(color='#7a8fa6')),
    margin=dict(l=16, r=16, t=32, b=16),
)

# ── Sidebar ──────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 EduMetrics")
    st.markdown("<p style='color:#7a8fa6;font-size:13px;'>Student Performance Analytics</p>", unsafe_allow_html=True)
    st.divider()

    st.markdown("<p style='color:#7a8fa6;font-size:11px;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;'>Filter Data</p>", unsafe_allow_html=True)

    attendance_range = st.slider("Rentang Kehadiran (%)", 60, 100, (60, 100))
    hours_range = st.slider("Jam Belajar/Minggu", 1, 44, (1, 44))

    st.divider()
    st.markdown("<p style='color:#7a8fa6;font-size:11px;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;'>Info Model</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#63b3ed;font-size:13px;'>Linear Regression</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#7a8fa6;font-size:12px;'>R² Score: <span style='color:#68d391;font-family:DM Mono'>{r2:.3f}</span></p>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#7a8fa6;font-size:12px;'>MAE: <span style='color:#68d391;font-family:DM Mono'>{mae:.3f}</span></p>", unsafe_allow_html=True)

# Filter data
df_filtered = df[
    (df['Attendance'] >= attendance_range[0]) &
    (df['Attendance'] <= attendance_range[1]) &
    (df['Hours_Studied'] >= hours_range[0]) &
    (df['Hours_Studied'] <= hours_range[1])
]

# ── Header ───────────────────────────────────────────
st.markdown("""
<div style='padding: 8px 0 32px 0;'>
    <p style='color:#63b3ed;font-size:12px;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;margin:0;'>MACHINE LEARNING DASHBOARD</p>
    <h1 style='font-size:36px;font-weight:600;color:#e8f4fd;margin:4px 0 8px 0;letter-spacing:-0.02em;'>Student Performance Analytics</h1>
    <p style='color:#7a8fa6;font-size:14px;margin:0;'>Analisis prediktif nilai ujian menggunakan Linear Regression · {total:,} siswa teranalisis</p>
</div>
""".format(total=len(df_filtered)), unsafe_allow_html=True)

# ── KPI Metrics ───────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Total Siswa", f"{len(df_filtered):,}")
with c2:
    st.metric("Rata-rata Nilai", f"{df_filtered['Exam_Score'].mean():.1f}")
with c3:
    st.metric("Nilai Tertinggi", f"{df_filtered['Exam_Score'].max():.0f}")
with c4:
    st.metric("R² Score", f"{r2:.3f}", delta="Model Accuracy")
with c5:
    st.metric("MAE", f"{mae:.2f}", delta="Avg Error")

st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)

# ── Charts Row 1 ──────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class='section-card'>
        <p class='section-title'>Analisis Korelasi</p>
        <p class='section-heading'>Jam Belajar vs Nilai Ujian</p>
    """, unsafe_allow_html=True)

    fig1 = px.scatter(df_filtered,
        x='Hours_Studied', y='Exam_Score',
        trendline='ols',
        color_discrete_sequence=['#63b3ed'],
        trendline_color_override='#fc8181')
    fig1.update_traces(marker=dict(size=4, opacity=0.4))
    fig1.update_layout(**PLOTLY_THEME, height=300,
        xaxis_title="Jam Belajar / Minggu",
        yaxis_title="Nilai Ujian")
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='section-card'>
        <p class='section-title'>Analisis Korelasi</p>
        <p class='section-heading'>Kehadiran vs Nilai Ujian</p>
    """, unsafe_allow_html=True)

    fig2 = px.scatter(df_filtered,
        x='Attendance', y='Exam_Score',
        trendline='ols',
        color_discrete_sequence=['#9f7aea'],
        trendline_color_override='#fc8181')
    fig2.update_traces(marker=dict(size=4, opacity=0.4))
    fig2.update_layout(**PLOTLY_THEME, height=300,
        xaxis_title="Kehadiran (%)",
        yaxis_title="Nilai Ujian")
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── Charts Row 2 ──────────────────────────────────────
col3, col4 = st.columns([1.2, 0.8])

with col3:
    st.markdown("""
    <div class='section-card'>
        <p class='section-title'>Feature Analysis</p>
        <p class='section-heading'>Korelasi Faktor terhadap Nilai</p>
    """, unsafe_allow_html=True)

    df_enc = df_filtered.copy()
    le2 = LabelEncoder()
    for k in df_enc.select_dtypes(include='object').columns:
        df_enc[k] = le2.fit_transform(df_enc[k])
    korelasi = df_enc.corr()['Exam_Score'].drop('Exam_Score').sort_values()

    colors = ['#fc8181' if v < 0 else '#63b3ed' for v in korelasi.values]
    fig3 = go.Figure(go.Bar(
        x=korelasi.values,
        y=korelasi.index,
        orientation='h',
        marker_color=colors,
        marker_line_width=0,
    ))
    fig3.update_layout(**PLOTLY_THEME, height=380,
        xaxis_title="Nilai Korelasi",
        yaxis_title="")
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class='section-card'>
        <p class='section-title'>Distribusi</p>
        <p class='section-heading'>Sebaran Nilai Ujian</p>
    """, unsafe_allow_html=True)

    fig4 = px.histogram(df_filtered, x='Exam_Score', nbins=25,
        color_discrete_sequence=['#63b3ed'])
    fig4.update_traces(marker_line_width=0, opacity=0.8)
    fig4.update_layout(**PLOTLY_THEME, height=380,
        xaxis_title="Nilai Ujian",
        yaxis_title="Jumlah Siswa",
        bargap=0.05)
    fig4.add_vline(x=df_filtered['Exam_Score'].mean(),
        line_dash="dash", line_color="#fc8181",
        annotation_text=f"Rata-rata: {df_filtered['Exam_Score'].mean():.1f}",
        annotation_font_color="#fc8181")
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── Tren Line ─────────────────────────────────────────
st.markdown("""
<div class='section-card'>
    <p class='section-title'>Trend Analysis</p>
    <p class='section-heading'>Rata-rata Nilai per Jam Belajar</p>
""", unsafe_allow_html=True)

tren = df_filtered.groupby('Hours_Studied')['Exam_Score'].agg(['mean', 'count']).reset_index()
fig5 = go.Figure()
fig5.add_trace(go.Scatter(
    x=tren['Hours_Studied'], y=tren['mean'],
    mode='lines+markers',
    line=dict(color='#63b3ed', width=2.5),
    marker=dict(color='#63b3ed', size=6),
    fill='tozeroy',
    fillcolor='rgba(99,179,237,0.08)',
    name='Rata-rata Nilai'
))
fig5.update_layout(**PLOTLY_THEME, height=280,
    xaxis_title="Jam Belajar / Minggu",
    yaxis_title="Rata-rata Nilai Ujian")
st.plotly_chart(fig5, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ── Prediction Section ────────────────────────────────
st.markdown("""
<div class='section-card'>
    <p class='section-title'>ML Prediction Engine</p>
    <p class='section-heading'>Simulasi Prediksi Nilai Ujian</p>
    <p style='color:#7a8fa6;font-size:13px;margin-bottom:24px;'>Sesuaikan parameter untuk memprediksi nilai ujian siswa secara real-time</p>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    jam_belajar = st.slider("⏱ Jam Belajar / Minggu", 1, 44, 20)
with col2:
    kehadiran = st.slider("📅 Kehadiran (%)", 60, 100, 80)
with col3:
    nilai_sebelumnya = st.slider("📝 Nilai Sebelumnya", 50, 100, 70)
with col4:
    sesi_les = st.slider("📚 Sesi Les", 0, 10, 2)

# Prediksi
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
avg = df['Exam_Score'].mean()
delta = hasil - avg

if hasil >= 80:
    badge = "<span class='badge badge-good'>✓ Di atas rata-rata</span>"
elif hasil >= 70:
    badge = "<span class='badge badge-avg'>~ Rata-rata</span>"
else:
    badge = "<span class='badge badge-low'>↓ Di bawah rata-rata</span>"

st.markdown(f"""
<div class='prediction-result'>
    <p class='prediction-label'>Prediksi Nilai Ujian</p>
    <p class='prediction-score'>{hasil:.1f}</p>
    <p style='color:#7a8fa6;font-size:13px;margin:4px 0;'>
        {'▲' if delta > 0 else '▼'} {abs(delta):.1f} poin dari rata-rata ({avg:.1f})
    </p>
    {badge}
</div>
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:32px 0 16px 0;'>
    <p style='color:#3a4a5c;font-size:12px;'>EduMetrics · Dibangun dengan Python, scikit-learn & Streamlit</p>
</div>
""", unsafe_allow_html=True)