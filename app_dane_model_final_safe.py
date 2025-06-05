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

# 🔄 Przygotuj dane wejściowe do predykcji
X = pd.get_dummies(df[['Stacja']], drop_first=False)

# Dodaj brakujące kolumny zgodnie z modelem
for col in model.feature_name_:
    if col not in X.columns:
        X[col] = 0

# Ustaw kolejność kolumn jak w modelu
X = X[model.feature_name_]

# 🧠 Predykcja
df['Predykcja awarii'] = model.predict(X)
df['Predykcja awarii'] = df['Predykcja awarii'].map({0: "🟢 Brak", 1: "🔴 Będzie"})

# 🎚️ Interfejs użytkownika
st.subheader("📋 Lista stacji z predykcją")

# ➤ Wybór tylko 1 daty – "jutro"
ostatnia_data = df['data_dzienna'].max()
wybrana_data = st.selectbox("📅 Dzień", options=[ostatnia_data], format_func=lambda x: "Jutro")

# ➤ Lista unikalnych linii
linie = sorted(df['Linia'].unique())
wybrana_linia = st.selectbox("🏭 Wybierz linię", linie)

# ➤ Filtrowanie
df_filtered = df[(df['data_dzienna'] == wybrana_data) & (df['Linia'] == wybrana_linia)].copy()

# ➤ Dodaj kolumnę LP
df_filtered.reset_index(drop=True, inplace=True)
df_filtered.index += 1
df_filtered.insert(0, "LP", df_filtered.index)

# ➤ Tabela
st.dataframe(
    df_filtered[['LP', 'data_dzienna', 'Linia', 'Stacja', 'Predykcja awarii']]
    .sort_values(by='Stacja'),
    use_container_width=True
)

# 💾 Eksport CSV
st.download_button(
    label="⬇️ Pobierz dane do CSV",
    data=df_filtered.to_csv(index=False).encode('utf-8'),
    file_name="predykcja_1dzien.csv",
    mime="text/csv"
)
