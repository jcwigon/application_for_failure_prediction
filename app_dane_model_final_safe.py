import streamlit as st
import pandas as pd
import joblib
from io import BytesIO

# 🎛️ Konfiguracja strony
st.set_page_config(page_title="Predykcja awarii", page_icon="🛠", layout="wide")

# 🧠 Tytuł
st.title("🛠 Predykcja awarii – 1 dzień do przodu")
st.info("Aplikacja przewiduje, czy jutro wystąpi awaria na stacji, na podstawie danych historycznych.")

# 📦 Wczytaj model
model = joblib.load("model_predykcji_awarii_lightgbm.pkl")

# 📊 Wczytaj dane do predykcji
df = pd.read_csv("dane_predykcja_1dzien.csv")
df['data_dzienna'] = pd.to_datetime(df['data_dzienna'])
df['Linia'] = df['Stacja'].str.extract(r'(DB\d{2}|DD\d{2}|DE\d{2}|DO\d{2}|PB\d{2})')

# ✳️ Przygotuj cechy
X = df[['Stacja']]
X_encoded = pd.get_dummies(X, drop_first=False)

# 🔄 Upewnij się, że kolumny zgadzają się z modelem
expected_cols = model.feature_name_
for col in expected_cols:
    if col not in X_encoded.columns:
        X_encoded[col] = 0
X_encoded = X_encoded[expected_cols]

# 🔮 Predykcja
df['Predykcja awarii'] = model.predict(X_encoded)
df['Predykcja awarii'] = df['Predykcja awarii'].map({0: "🟢 Brak", 1: "🔴 Będzie"})

# 📋 Interfejs użytkownika
st.subheader("📋 Lista stacji z predykcją")

# 🔘 Jutro jako data
jutro = df['data_dzienna'].max()
st.markdown(f"**Dzień:** {jutro.strftime('%Y-%m-%d')} (Jutro)")

# 🔽 Wybór linii
linie = sorted(df['Linia'].astype(str).unique())
wybrana_linia = st.selectbox("🏭 Wybierz linię", linie)

# 🔍 Filtrowanie
df_filtered = df[df['Linia'] == wybrana_linia].copy()
df_filtered = df_filtered.sort_values(by="Stacja")
df_filtered.insert(0, "Lp.", range(1, len(df_filtered) + 1))

# 📊 Metryka
liczba_awarii = (df_filtered['Predykcja awarii'] == "🔴 Będzie").sum()
st.metric(label="🔧 Przewidywane awarie", value=f"{liczba_awarii} stacji")

# 🧾 Tabela
st.dataframe(
    df_filtered[['Lp.', 'Stacja', 'Predykcja awarii']],
    use_container_width=True
)

# 📤 Eksport do CSV
st.download_button(
    label="⬇️ Pobierz dane do CSV",
    data=df_filtered.to_csv(index=False).encode('utf-8'),
    file_name="predykcja_wyniki.csv",
    mime="text/csv"
)

