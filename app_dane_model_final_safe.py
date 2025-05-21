
import streamlit as st
import pandas as pd
import joblib
from io import BytesIO

# 🎛️ Konfiguracja strony
st.set_page_config(page_title="Predykcja awarii", page_icon="🛠", layout="wide")

# 🧠 Tytuł i opis
st.title("🛠 Predykcja awarii – 3 dni do przodu")
st.info("Ta aplikacja wykorzystuje model ML do przewidywania awarii maszyn na podstawie danych historycznych.")

# 📦 Wczytaj model
model = joblib.load("model_predykcji_awarii_lightgbm.pkl")

# 📊 Wczytaj dane do predykcji
df = pd.read_csv("dane_predykcja_model.csv")
df['data_dzienna'] = pd.to_datetime(df['data_dzienna'])

# 🧾 Przygotuj dane
features = ['awarie_7dni', 'dni_od_ostatniej_awarii', 'zmiana', 'Linia', 'Stacja']
df_encoded = pd.get_dummies(df[features], drop_first=True)

# 🔄 Upewnij się, że kolumny są zgodne z modelem
expected_cols = model.feature_name_
for col in expected_cols:
    if col not in df_encoded.columns:
        df_encoded[col] = 0
df_encoded = df_encoded[expected_cols]

# 🛡️ Zabezpieczenie przed pustym DataFrame
if df_encoded.empty:
    st.warning("Brak danych do predykcji. Sprawdź dane wejściowe lub wybierz inne filtry.")
    st.stop()

# 🤖 Predykcja
df['Predykcja awarii'] = model.predict(df_encoded)
df['Predykcja awarii'] = df['Predykcja awarii'].map({0: "🟢 Brak", 1: "🔴 Będzie"})

# 📋 Interfejs użytkownika
st.subheader("📋 Lista stacji z predykcją")

# Filtry
unikalne_daty = sorted(df['data_dzienna'].unique())
unikalne_linie = sorted(df['Linia'].unique())

wybrana_data = st.selectbox("📅 Wybierz dzień", unikalne_daty)
wybrana_linia = st.selectbox("🏭 Wybierz linię", ["Wszystkie"] + unikalne_linie)

# Filtrowanie
df_filtered = df[df['data_dzienna'] == pd.to_datetime(wybrana_data)]
if wybrana_linia != "Wszystkie":
    df_filtered = df_filtered[df_filtered['Linia'] == wybrana_linia]

# 📊 Metryka
liczba_awarii = (df_filtered['Predykcja awarii'] == '🔴 Będzie').sum()
st.metric(label="🔧 Przewidywane awarie", value=f"{liczba_awarii} stacji")

# Tabela z kolorem
st.dataframe(
    df_filtered[['data_dzienna', 'Linia', 'Stacja', 'Predykcja awarii']]
    .sort_values(by='Predykcja awarii', ascending=False),
    use_container_width=True
)

# Eksport CSV
st.download_button(
    label="⬇️ Pobierz widoczne dane do CSV",
    data=df_filtered.to_csv(index=False).encode('utf-8'),
    file_name="predykcja_wyniki.csv",
    mime="text/csv"
)

# Eksport XLSX
def convert_df_to_excel_bytes(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name="Predykcja")
    return output.getvalue()

excel_data = convert_df_to_excel_bytes(df_filtered)

st.download_button(
    label="⬇️ Pobierz widoczne dane do Excel (XLSX)",
    data=excel_data,
    file_name="predykcja_wyniki.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
