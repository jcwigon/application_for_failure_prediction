import streamlit as st
import pandas as pd
import joblib

# Konfiguracja strony
st.set_page_config(page_title="Predykcja awarii", page_icon="🛠", layout="wide")

# Tytuł i opis
st.title("🛠 Predykcja awarii – 1 dzień do przodu")
st.info("Aplikacja przewiduje awarie maszyn na podstawie danych z przeszłości.")

# Wczytaj model
model = joblib.load("model_predykcji_awarii_lightgbm.pkl")

# Wczytaj dane predykcyjne
df = pd.read_csv("dane_predykcja_1dzien.csv")
df['data_dzienna'] = pd.to_datetime(df['data_dzienna'])

# Zakoduj dane wejściowe zgodnie z modelem
X = pd.get_dummies(df[['Stacja']])
for col in model.feature_name_:
    if col not in X.columns:
        X[col] = 0
X = X[model.feature_name_]

# Predykcja
df['Predykcja awarii'] = model.predict(X)
df['Predykcja awarii'] = df['Predykcja awarii'].map({0: "🟢 Brak", 1: "🔴 Będzie"})

# Wydzielenie informacji o linii z nazwy stacji
df['Linia'] = df['Stacja'].str.extract(r'^(DB\d{2})')

# Lista dostępnych dni i linii
unikalne_daty = sorted(df['data_dzienna'].unique())
unikalne_linie = sorted(df['Linia'].dropna().unique())

# Filtry użytkownika
wybrana_data = st.selectbox("📅 Wybierz dzień", unikalne_daty)
wybrana_linia = st.selectbox("🏭 Wybierz linię", unikalne_linie)

# Filtrowanie danych
df_filtered = df[(df['data_dzienna'] == wybrana_data) & (df['Linia'] == wybrana_linia)]

# Liczba awarii
liczba_awarii = (df_filtered['Predykcja awarii'] == '🔴 Będzie').sum()
st.metric(label="🔧 Przewidywane awarie", value=f"{liczba_awarii} stacji")

# Tabela wyników
st.dataframe(
    df_filtered[['data_dzienna', 'Linia', 'Stacja', 'Predykcja awarii']]
    .sort_values(by='Predykcja awarii', ascending=False),
    use_container_width=True
)

# Eksport CSV
st.download_button(
    label="⬇️ Pobierz dane do CSV",
    data=df_filtered.to_csv(index=False).encode('utf-8'),
    file_name="predykcja_1dzien.csv",
    mime="text/csv"
)
