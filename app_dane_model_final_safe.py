import streamlit as st
import pandas as pd
import joblib
from io import BytesIO

st.set_page_config(page_title="Predykcja awarii", page_icon="🛠", layout="wide")

st.title("🛠 Predykcja awarii – 1 dzień do przodu")
st.info("Aplikacja przewiduje, czy jutro wystąpi awaria na stacji, na podstawie danych historycznych.")

# 📦 Wczytaj model
model = joblib.load("model_predykcji_awarii_lightgbm.pkl")

# 📊 Wczytaj dane
df = pd.read_csv("dane_predykcja_1dzien.csv")
df['data_dzienna'] = pd.to_datetime(df['data_dzienna'])

# ⏳ Ustal tylko jeden dzień – jutro
data_jutra = df['data_dzienna'].max()
st.markdown(f"**Dzień:** {data_jutra.date()} (Jutro)")

# 📍 Filtr linii
linie = sorted(df['Stacja'].str.extract(r'(^[A-Z]{2,}[0-9]{2,})')[0].dropna().unique())
wybrana_linia = st.selectbox("🏭 Wybierz linię", linie)

# 🔢 Przygotowanie danych
X = df[['Stacja']]
X['Stacja'] = X['Stacja'].astype(str)
X_encoded = pd.get_dummies(X, drop_first=False)

# 🧠 Predykcja
df['Predykcja awarii'] = model.predict(X_encoded)
df['Predykcja awarii'] = df['Predykcja awarii'].map({0: "🟢 Brak", 1: "🔴 Będzie"})

# 🔍 Filtrowanie tylko dla jutra i wybranej linii
df_filtered = df[df['data_dzienna'] == data_jutra].copy()
df_filtered = df_filtered[df_filtered['Stacja'].str.startswith(wybrana_linia)]

# 🧹 Usuń duplikaty stacji
df_filtered = df_filtered.drop_duplicates(subset=['Stacja'])

# 🧾 Dodaj kolumnę Linia
if 'Linia' in df_filtered.columns:
    df_filtered.drop(columns=['Linia'], inplace=True)
df_filtered.insert(1, "Linia", wybrana_linia)

# 🔢 Dodaj Lp
df_filtered.insert(0, "Lp.", range(1, len(df_filtered) + 1))

# 📋 Wyświetl metrykę
liczba_awarii = (df_filtered['Predykcja awarii'] == '🔴 Będzie').sum()
st.metric(label="🔧 Przewidywane awarie", value=f"{liczba_awarii} stacji")

# 📊 Tabela wyników
st.dataframe(
    df_filtered[['Lp.', 'Linia', 'Stacja', 'Predykcja awarii']].reset_index(drop=True),
    use_container_width=True
)

# 💾 Eksport CSV
st.download_button(
    label="⬇️ Pobierz dane do CSV",
    data=df_filtered.to_csv(index=False).encode('utf-8'),
    file_name="predykcja_1dzien.csv",
    mime="text/csv"
)


