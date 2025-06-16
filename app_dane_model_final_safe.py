import streamlit as st
import pandas as pd
import joblib
import re
from io import BytesIO

st.set_page_config(page_title="Predykcja awarii + Konwerter", page_icon="🛠", layout="wide")

st.title("🛠 Predykcja awarii – z konwerterem pliku Excel")
st.info("Wgraj plik **DispatchHistory--*.xlsx**, aplikacja go przekształci i przewidzi awarie.")

# 📦 Wczytaj model predykcji
model = joblib.load("model_predykcji_awarii_lightgbm.pkl")

def extract_date_from_filename(filename):
    match = re.search(r'DispatchHistory--(\d{4}-\d{2}-\d{2})', filename)
    return match.group(1) if match else None

def convert_excel_to_model_input(file, filename):
    # Wczytaj Excel
    df = pd.read_excel(file)

    # Sprawdzenie wymaganych kolumn
    required_cols = ['machinecode', 'linecode']
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Brak wymaganej kolumny '{col}' w pliku.")
            return None

    # Parsowanie daty z nazwy pliku
    data_dzienna = extract_date_from_filename(filename)
    if not data_dzienna:
        st.error("Nie udało się wyciągnąć daty z nazwy pliku. Upewnij się, że plik ma nazwę np. DispatchHistory--2025-05-26.xlsx")
        return None

    # Filtrowanie rekordów poprawnych
    df = df.dropna(subset=['machinecode', 'linecode'])

    # Budujemy unikalne zgłoszenia awarii
    df_awarie = df[['machinecode', 'linecode']].drop_duplicates()
    df_awarie = df_awarie.rename(columns={
        'machinecode': 'Stacja',
        'linecode': 'Linia'
    })
    df_awarie['data_dzienna'] = data_dzienna
    df_awarie['czy_wystapila_awaria'] = 1

    return df_awarie[['data_dzienna', 'Stacja', 'Linia', 'czy_wystapila_awaria']]

uploaded_file = st.file_uploader("📤 Wgraj plik Excel (DispatchHistory--*.xlsx)", type=['xlsx'])

if uploaded_file is not None:
    with st.spinner("⏳ Przetwarzanie pliku..."):
        converted_df = convert_excel_to_model_input(uploaded_file, uploaded_file.name)

        if converted_df is not None:
            st.success("✅ Plik poprawnie przekształcony. Oto dane wejściowe dla modelu:")
            st.dataframe(converted_df)

            # 📊 Predykcja
            X_pred = converted_df[['data_dzienna', 'Stacja', 'Linia']]
            # Kodowanie cech (jeśli model tego wymaga - np. one-hot lub label encoding)
            # UWAGA: Musisz dopasować poniżej do sposobu trenowania modelu
            # Poniżej zakładam, że model oczekuje tylko zakodowanej kolumny 'Stacja'

            # Przygotowanie danych do predykcji – placeholder (dopasuj do swojego modelu)
            X_pred_encoded = pd.get_dummies(X_pred['Stacja'])  # przykład – zmień według potrzeb
            missing_cols = [col for col in model.feature_names_in_ if col not in X_pred_encoded.columns]
            for col in missing_cols:
                X_pred_encoded[col] = 0
            X_pred_encoded = X_pred_encoded[model.feature_names_in_]

            y_pred = model.predict(X_pred_encoded)

            converted_df['Predykcja_awarii'] = y_pred
            st.subheader("🔎 Wynik predykcji")
            st.dataframe(converted_df)

            # 📥 Pobranie wyniku jako CSV
            csv = converted_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Pobierz wynik jako CSV", data=csv, file_name="wynik_predykcji.csv", mime='text/csv')


