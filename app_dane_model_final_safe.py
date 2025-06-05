import streamlit as st
import pandas as pd
import joblib
from io import BytesIO

# 🎛️ Konfiguracja strony
st.set_page_config(page_title="Predykcja awarii – 1 dzień do przodu", page_icon="🛠", layout="wide")

# 🧠 Tytuł i opis
st.title("🛠 Predykcja awarii – 1 dzień do przodu")
st.info("Aplikacja przewiduje awarie maszyn na podstawie danych z przeszłości.")

# 📦 Wczytaj model
model = joblib.load("model_predykcji_awarii_lightgbm.pkl")

# 📊 Wczytaj dane
df = pd.read_csv("dane_predykcja_1dzien.csv")
df['data_dzienna'] = pd.to_datetime(df['data_dzienna'])

# 🤖 Przygotuj predykcję
X = df[['Stacja']]
df['Predykcja awarii'] = model.predict(X)
df['Predykcja awarii'] = df['Predykcja awarii'].map({0: "🟢 Brak", 1: "🔴 Będzie"})

# 📅 Stały wybór – „Jutro”
st.subheader("📋 Lista stacji z predykcją")
st.selectbox("📅 Dzień", ["Jutro"])

# 🏭 Filtracja linii (pełna lista z danych)
unikalne_linie = sorted(df['Stacja'].str.extract(r"^(DB\d{2})")[0].dropna().unique())
wybrana_linia = st.selectbox("🏭 Wybierz linię", unikalne_linie)

# 🔍 Filtrowanie stacji
df_filtered = df[df['Stacja'].str.startswith(wybrana_linia)].copy()

# 🔢 Dodaj numerację od 1
df_filtered.reset_index(drop=True, inplace=True)
df_filtered.index += 1

# 📋 Tabela
df_filtered = df_filtered.rename(columns={"data_dzienna": "Dzień"})
st.dataframe(
    df_filtered[['Dzień', 'Stacja', 'Predykcja awarii']],
    use_container_width=True
)

# 📁 Eksport CSV
st.download_button(
    label="⬇️ Pobierz dane do CSV",
    data=df_filtered.to_csv(index_label="Lp.", encoding="utf-8").encode('utf-8'),
    file_name="predykcja_awarii_jutro.csv",
    mime="text/csv"
)
