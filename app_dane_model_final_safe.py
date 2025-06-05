import streamlit as st
import pandas as pd
import joblib
from io import BytesIO

# 🎛️ Konfiguracja strony
st.set_page_config(page_title="Predykcja awarii – 1 dzień do przodu", page_icon="🛠", layout="wide")

# 🧠 Tytuł i nagłówek
st.title("🛠 Predykcja awarii – 1 dzień do przodu")
st.info("Aplikacja przewiduje, czy jutro wystąpi awaria na stacji, na podstawie danych historycznych.")

# 📦 Wczytaj model i dane
model = joblib.load("model_predykcji_awarii_lightgbm.pkl")
df = pd.read_csv("dane_predykcja_1dzien.csv")
df['data_dzienna'] = pd.to_datetime(df['data_dzienna'])

# 🧾 Przygotowanie danych do predykcji
X = pd.get_dummies(df[["Stacja"]], drop_first=True)

# 🔮 Predykcja
df["Predykcja awarii"] = model.predict(X)
df["Predykcja awarii"] = df["Predykcja awarii"].map({0: "🟢 Brak", 1: "🔴 Będzie"})

# 📆 Ustal dzień jutro
st.subheader("📋 Lista stacji z predykcją")
st.markdown("**Dzień:** Jutro")

# 📍 Filtr linii
unikalne_linie = sorted(df["Linia"].unique())
wybrana_linia = st.selectbox("🏭 Wybierz linię", unikalne_linie)

# 🔎 Filtrowanie
df_filtered = df[df["Linia"] == wybrana_linia].copy()
df_filtered = df_filtered.groupby("Stacja", as_index=False).first()

# 🔢 Dodanie LP
df_filtered.insert(0, "Lp.", range(1, len(df_filtered) + 1))

# 🛠️ Tabela
st.metric(label="🔧 Przewidywane awarie", value=f"{(df_filtered['Predykcja awarii'] == '🔴 Będzie').sum()} stacji")

st.dataframe(
    df_filtered[["Lp.", "Linia", "Stacja", "Predykcja awarii"]].reset_index(drop=True),
    use_container_width=True
)

# 💾 Eksport do CSV
st.download_button(
    label="⬇️ Pobierz dane do CSV",
    data=df_filtered.to_csv(index=False).encode("utf-8"),
    file_name="predykcja_1dzien.csv",
    mime="text/csv"
)

# 💾 Eksport do XLSX
def convert_df_to_excel_bytes(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Predykcja")
    return output.getvalue()

excel_data = convert_df_to_excel_bytes(df_filtered)

st.download_button(
    label="⬇️ Pobierz dane do Excel (XLSX)",
    data=excel_data,
    file_name="predykcja_1dzien.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

