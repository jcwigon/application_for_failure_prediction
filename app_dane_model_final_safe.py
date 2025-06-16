import streamlit as st
import pandas as pd
import joblib
import re
from io import BytesIO

st.set_page_config(page_title="Predykcja awarii", page_icon="🛠", layout="wide")

st.title("🛠 Predykcja awarii – 1 dzień do przodu")
st.info("Wgraj plik DispatchHistory--*.csv, aplikacja go przekształci i przewidzi awarie.")

# 📦 Wczytaj model
try:
    model = joblib.load("model_predykcji_awarii_lightgbm.pkl")
except Exception as e:
    st.error(f"Błąd podczas wczytywania modelu: {str(e)}")
    st.stop()

def extract_date_from_filename(filename):
    match = re.search(r'DispatchHistory--(\d{4}-\d{2}-\d{2})', filename)
    return match.group(1) if match else None

def convert_csv_to_model_input(file, filename):
    try:
        # Wczytaj CSV z różnymi opcjami kodowania i separatorami
        for encoding in ['utf-8', 'latin1', 'cp1250']:
            try:
                file.seek(0)
                df = pd.read_csv(file, header=0, encoding=encoding, sep=';')
                if len(df.columns) > 1:
                    break
            except:
                continue
        else:
            st.error("Nie można odczytać pliku - problem z kodowaniem znaków lub separatorem")
            return None

        # Sprawdzenie wymaganych kolumn
        df.columns = df.columns.str.strip().str.lower()
        required_cols = ['machinecode', 'linecode']
        
        found_machinecode = [col for col in df.columns if 'machinecode' in col]
        found_linecode = [col for col in df.columns if 'linecode' in col]
        
        if not found_machinecode or not found_linecode:
            st.error(f"Nie znaleziono wymaganych kolumn zawierających 'machinecode' i 'linecode'")
            return None
        
        machinecode_col = found_machinecode[0]
        linecode_col = found_linecode[0]

        # Parsowanie daty z nazwy pliku
        data_dzienna = extract_date_from_filename(filename)
        if not data_dzienna:
            st.error("Nie udało się wyciągnąć daty z nazwy pliku. Wymagany format: DispatchHistory--RRRR-MM-DD.csv")
            return None

        # Filtrowanie i czyszczenie danych
        df = df.dropna(subset=[machinecode_col, linecode_col])
        df['machinecode_clean'] = df[machinecode_col].astype(str).str.extract(r'([A-Za-z0-9]+)')[0]
        df['linecode_clean'] = df[linecode_col].astype(str).str.extract(r'([A-Za-z0-9]+)')[0]

        # Przygotowanie danych wyjściowych
        df_out = df[['machinecode_clean', 'linecode_clean']].drop_duplicates()
        df_out = df_out.rename(columns={
            'machinecode_clean': 'Stacja',
            'linecode_clean': 'Linia'
        })
        df_out['data_dzienna'] = pd.to_datetime(data_dzienna) + pd.Timedelta(days=1)  # Jutro
        
        return df_out

    except Exception as e:
        st.error(f"Błąd podczas przetwarzania pliku: {str(e)}")
        return None

# UI do przesyłania plików
uploaded_file = st.file_uploader("📤 Wgraj plik CSV (DispatchHistory--*.csv)", type=['csv'])

if uploaded_file is not None:
    with st.spinner("⏳ Przetwarzanie pliku..."):
        converted_df = convert_csv_to_model_input(uploaded_file, uploaded_file.name)

        if converted_df is not None:
            st.success("✅ Plik poprawnie przekształcony")
            
            # ⏳ Dzień jutro – tylko jako tekst
            st.markdown(f"**Dzień:** Jutro")

            # 📍 Filtr linii
            linie = sorted(converted_df['Linia'].unique())
            wybrana_linia = st.selectbox("🏭 Wybierz linię", linie)

            # 🔢 Przygotowanie danych
            X = converted_df[['Stacja']].copy()
            X['Stacja'] = X['Stacja'].astype(str)
            X_encoded = pd.get_dummies(X, drop_first=False)

            # 🧠 Predykcja
            converted_df['Predykcja awarii'] = model.predict(X_encoded)
            converted_df['Predykcja awarii'] = converted_df['Predykcja awarii'].map({0: "🟢 Brak", 1: "🔴 Będzie"})

            # 🔍 Filtrowanie tylko dla wybranej linii
            df_filtered = converted_df[converted_df['Linia'] == wybrana_linia].copy()

            # 🧹 Usuń duplikaty stacji
            df_filtered = df_filtered.drop_duplicates(subset=['Stacja'])

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

            # 💾 Eksport XLSX
            def convert_df_to_excel_bytes(df):
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name="Predykcja")
                return output.getvalue()

            st.download_button(
                label="⬇️ Pobierz dane do Excel (XLSX)",
                data=convert_df_to_excel_bytes(df_filtered),
                file_name="predykcja_1dzien.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
