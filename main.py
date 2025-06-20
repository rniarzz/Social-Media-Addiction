import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.set_page_config(page_title="Prediksi Kecanduan Media Sosial", layout="centered")

# === MAIN CONTENT ===
st.markdown("""
# 📱 Prediksi Tingkat Kecanduan Media Sosial Mahasiswa

### 🧠 Tentang Model Prediksi

Model yang digunakan untuk memprediksi tingkat kecanduan media sosial mahasiswa adalah *Regresi Linear*. Model ini menganalisis seberapa besar pengaruh setiap faktor (usia, jenis kelamin, jenjang pendidikan, platform media sosial yang paling sering digunakan, rata-rata jam penggunaan per hari, jam tidur, skor kesehatan mental, status hubungan, dan jumlah konflik akibat media sosial) terhadap skor kecanduan yang diperoleh dari data riil mahasiswa.

Setiap fitur memiliki bobot (koefisien) yang menunjukkan *seberapa besar kontribusinya* terhadap skor kecanduan. Semakin besar nilai koefisien (positif/negatif), semakin besar pula pengaruh faktor tersebut dalam menentukan tingkat kecanduan.
""")

# === LOAD & PREPROCESS DATA ===
@st.cache_data
def load_data():
    df = pd.read_csv("student.csv", sep=";")
    df["Avg_Daily_Usage_Hours"] = df["Avg_Daily_Usage_Hours"].astype(str).str.replace(",", ".").astype(float)
    df["Sleep_Hours_Per_Night"] = df["Sleep_Hours_Per_Night"].astype(str).str.replace(",", ".").astype(float)
    return df

df = load_data()
fitur = [
    "Age", "Gender", "Academic_Level", "Avg_Daily_Usage_Hours",
    "Most_Used_Platform", "Sleep_Hours_Per_Night", "Mental_Health_Score",
    "Relationship_Status", "Conflicts_Over_Social_Media"
]
target = "Addicted_Score"

# Encode kategorikal
df_enc = df.copy()
le_dict = {}
for kolom in ["Gender", "Academic_Level", "Most_Used_Platform", "Relationship_Status"]:
    le = LabelEncoder()
    df_enc[kolom] = le.fit_transform(df_enc[kolom])
    le_dict[kolom] = le

# Model training
X = df_enc[fitur]
y = df_enc[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# === TABEL FAKTOR ===
# Hitung rata-rata fitur (numerik: mean, kategorikal: modus lalu encode)
rata2_fitur = []
for kolom in fitur:
    if kolom in le_dict:
        mode_val = df[kolom].mode()[0]
        encoded = le_dict[kolom].transform([mode_val])[0]
        rata2_fitur.append(encoded)
    else:
        rata2_fitur.append(df[kolom].mean())
rata2_fitur = np.array(rata2_fitur)
koef = model.coef_
kontribusi = rata2_fitur * koef

tabel = pd.DataFrame({
    "Faktor": [
        "Usia",
        "Jenis Kelamin (0=Laki-laki, 1=Perempuan)",
        "Jenjang Pendidikan",
        "Rata-rata Jam Penggunaan",
        "Platform Paling Sering",
        "Jam Tidur per Malam",
        "Skor Kesehatan Mental",
        "Status Hubungan",
        "Konflik karena Media Sosial"
    ],
    "Rata-rata Data": rata2_fitur,
    "Koefisien Model": koef,
    "Kontribusi": kontribusi
})

st.markdown("### 🧪 Faktor-Faktor Penentu Skor Kecanduan Media Sosial")
st.dataframe(tabel.style.format({
    "Rata-rata Data": "{:.2f}",
    "Koefisien Model": "{:.2f}",
    "Kontribusi": "{:.2f}"
}), use_container_width=True)

# === FORM INPUT ===
with st.form("form_prediksi"):
    st.header("Masukkan Data Mahasiswa")
    col1, col2 = st.columns(2)
    with col1:
        usia = st.number_input("Usia", min_value=int(df["Age"].min()), max_value=int(df["Age"].max()), value=20)
        gender = st.selectbox("Jenis Kelamin", le_dict["Gender"].classes_)
        tingkat = st.selectbox("Jenjang Pendidikan", le_dict["Academic_Level"].classes_)
        platform = st.selectbox("Platform Media Sosial Terbanyak", le_dict["Most_Used_Platform"].classes_)
        status = st.selectbox("Status Hubungan", le_dict["Relationship_Status"].classes_)
    with col2:
        penggunaan = st.slider("Rata-rata Penggunaan Media Sosial (jam/hari)", 0.0, 12.0, 5.0, 0.1)
        tidur = st.slider("Rata-rata Jam Tidur per Malam", 3.0, 10.0, 6.0, 0.1)
        mental = st.slider("Skor Kesehatan Mental (1=buruk, 10=baik)", 1, 10, 6, 1)
        konflik = st.slider("Jumlah Konflik karena Media Sosial", 0, 5, 2, 1)
    tombol = st.form_submit_button("Proses")

if tombol:
    input_df = pd.DataFrame({
        "Age": [usia],
        "Gender": le_dict["Gender"].transform([gender]),
        "Academic_Level": le_dict["Academic_Level"].transform([tingkat]),
        "Avg_Daily_Usage_Hours": [penggunaan],
        "Most_Used_Platform": le_dict["Most_Used_Platform"].transform([platform]),
        "Sleep_Hours_Per_Night": [tidur],
        "Mental_Health_Score": [mental],
        "Relationship_Status": le_dict["Relationship_Status"].transform([status]),
        "Conflicts_Over_Social_Media": [konflik]
    })
    input_scaled = scaler.transform(input_df)
    skor_prediksi = model.predict(input_scaled)[0]
    persen_kecanduan = np.clip((skor_prediksi / 10) * 100, 0, 100)

    st.subheader("Hasil Prediksi")
    st.write(f"*Skor Kecanduan (Addicted_Score): {skor_prediksi:.2f}*")
    st.write(f"*Persentase Kecanduan:* {persen_kecanduan:.1f}%")
    if skor_prediksi >= 8:
        st.error("Risiko Kecanduan Tinggi")
    elif skor_prediksi >= 6:
        st.warning("Risiko Kecanduan Sedang")
    else:
        st.success("Risiko Kecanduan Rendah")