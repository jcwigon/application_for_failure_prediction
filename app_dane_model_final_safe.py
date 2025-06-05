import streamlit as st
import pandas as pd
import joblib
from io import BytesIO

st.set_page_config(page_title="Predykcja awarii – 1 dzień do przodu", page_icon="🛠", layout="wide")

st.title("🛠 Predykcja awarii – 1 dzień do przodu")
st.info("Aplikacja przewiduje, czy jutro wystąpi awaria na stacji, na podstawie danych historycznych.")

# Wczytaj model i dane
model = joblib.load("model_predykcji_awarii_lightgbm.pkl")
df = pd.read_csv("dane_predykcja_1dzien.csv")
df['data_dzienna'] = pd.to_datetime(df['data_dzienna'])

# Ustal datę predykcji (jutro)
jutro = df['data_dzienna'].max()
st.subheader("📋 Lista stacji z predykcją")
st.markdown(f"**Dzień:** {jutro.date()} (Jutro)")

# Filtr: wybór linii
unikalne_linie = sorted(df['Stacja'].str.extract(r'(^[A-Z]+\d+)')[0].dropna().unique())
wybrana_linia = st.selectbox("🏭 Wybierz linię", unikalne_linie)

# Filtrowanie: tylko dane z jutra i wybranej linii
df_filtered = df[
    (df['data_dzienna'] == jutro) &
    (df['Stacja'].str.startswith(wybrana_linia))
].copy()

# Przygotuj dane do predykcji
X = df_filtered[['Stacja']]
X['Stacja'] = X['Stacja'].astype(str)
X_encoded = pd.get_dummies(X)

# Dopasuj kolumny do modelu
expected_cols = model.feature_name_
for col in expected_cols:
    if col not in X_encoded.columns:
        X_encoded[col] = 0
X_encoded = X_encoded[expected_cols]

# Predykcja
df_filtered['Predykcja awarii'] = model.predict(X_encoded)
df_filtered['Predykcja awarii'] = df_filtered['Predykcja awarii'].map({0: "🟢 Brak", 1: "🔴 Będzie"})

# Usuń duplikaty stacji – po jednej unikalnej stacji
df_filtered = df_filtered.drop_duplicates(subset='Stacja')

# Dodaj kolumny: Lp. i Linia
df_filtered.insert(0, "Lp.", range(1, len(df_filtered) + 1))
df_filtered.insert(1, "Linia", wybrana_linia)

# Wyświetl tabelę
st.metric(label="🔧 Przewidywane awarie", value=f"{(df_filtered['Predykcja awarii'] == '🔴 Będzie').sum()} stacji")
st.dataframe(
    df_filtered[['Lp.', 'Linia', 'Stacja', 'Predykcja awarii']].reset_index(drop=True),
    use_container_width=True
)

# Eksport CSV
st.download_button(
    label="⬇️ Pobierz dane do CSV",
    data=df_filtered.to_csv(index=False).encode('utf-8'),
    file_name="predykcja_jutro.csv",
    mime="text/csv"
)

