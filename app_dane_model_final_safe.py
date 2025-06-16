import streamlit as st
import pandas as pd
import joblib
import re
from io import BytesIO

st.set_page_config(page_title="Predykcja awarii + Konwerter CSV", page_icon="🛠", layout="wide")

st.title("🛠 Predykcja awarii – z konwerterem pliku CSV")
st.info("Wgraj plik **DispatchHistory--*.csv**, aplikacja go przekształci i przewidzi awarie.")

# 📦 Wczytaj model predykcji
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
                file.seek(0)  # Reset pozycji pliku przed ponownym odczytem
                # Próba wczytania z różnymi separatorami
                for sep in [';', ',', '\t']:
                    try:
                        df = pd.read_csv(file, header=0, encoding=encoding, sep=sep)
                        if len(df.columns) > 1:  # Jeśli udało się podzielić na kolumny
                            break
                    except:
                        continue
                else:
                    continue
                break
            except UnicodeDecodeError:
                continue
        else:
            st.error("Nie można odczytać pliku - problem z kodowaniem znaków lub separatorem")
            return None

        # Debugowanie - pokaż nagłówki
        st.write("Znalezione kolumny:", df.columns.tolist())
        
        # Sprawdzenie wymaganych kolumn (case-insensitive i z uwzględnieniem białych znaków)
        df.columns = df.columns.str.strip().str.lower()
        required_cols = ['machinecode', 'linecode']
        
        # Sprawdź czy wymagane kolumny istnieją (w całości lub jako część nazwy kolumny)
        found_machinecode = [col for col in df.columns if 'machinecode' in col]
        found_linecode = [col for col in df.columns if 'linecode' in col]
        
        if not found_machinecode or not found_linecode:
            st.error(f"Nie znaleziono wymaganych kolumn. Szukane kolumny zawierające: 'machinecode' i 'linecode'")
            st.error(f"Dostępne kolumny to: {df.columns.tolist()}")
            return None
        
        # Wybierz pierwsze pasujące kolumny
        machinecode_col = found_machinecode[0]
        linecode_col = found_linecode[0]

        # Parsowanie daty z nazwy pliku
        data_dzienna = extract_date_from_filename(filename)
        if not data_dzienna:
            st.error("Nie udało się wyciągnąć daty z nazwy pliku. Wymagany format: DispatchHistory--RRRR-MM-DD.csv")
            return None

        # Filtrowanie i czyszczenie danych
        df = df.dropna(subset=[machinecode_col, linecode_col])
        
        # Ekstrakcja podstawowych wartości
        df['machinecode_clean'] = df[machinecode_col].astype(str).str.extract(r'([A-Za-z0-9]+)')[0]
        df['linecode_clean'] = df[linecode_col].astype(str).str.extract(r'([A-Za-z0-9]+)')[0]

        # Budujemy unikalne zgłoszenia awarii
        df_awarie = df[['machinecode_clean', 'linecode_clean']].drop_duplicates()
        df_awarie = df_awarie.rename(columns={
            'machinecode_clean': 'Stacja',
            'linecode_clean': 'Linia'
        })
        df_awarie['data_dzienna'] = data_dzienna
        df_awarie['czy_wystapila_awaria'] = 1

        return df_awarie[['data_dzienna', 'Stacja', 'Linia', 'czy_wystapila_awaria']]

    except Exception as e:
        st.error(f"Błąd podczas przetwarzania pliku: {str(e)}")
        return None

uploaded_file = st.file_uploader("📤 Wgraj plik CSV (DispatchHistory--*.csv)", type=['csv'])

if uploaded_file is not None:
    with st.spinner("⏳ Przetwarzanie pliku..."):
        # Podgląd pierwszych linii pliku
        file_content = uploaded_file.getvalue().decode('utf-8', errors='replace').split('\n')
        st.text("Pierwsze linie pliku:\n" + "\n".join(file_content[:3]))
        
        converted_df = convert_csv_to_model_input(uploaded_file, uploaded_file.name)

        if converted_df is not None:
            st.success(f"✅ Plik poprawnie przekształcony. Liczba rekordów: {len(converted_df)}")
            st.dataframe(converted_df)

            # 📊 Predykcja
            try:
                X_pred = converted_df[['data_dzienna', 'Stacja', 'Linia']]
                
                # Kodowanie cech
                X_pred_encoded = pd.get_dummies(X_pred['Stacja'])
                
                # Upewnij się, że mamy wszystkie wymagane kolumny
                missing_cols = set(model.feature_names_in_) - set(X_pred_encoded.columns)
                for col in missing_cols:
                    X_pred_encoded[col] = 0
                
                X_pred_encoded = X_pred_encoded[model.feature_names_in_]
                y_pred = model.predict(X_pred_encoded)

                converted_df['Predykcja_awarii'] = y_pred
                
                # Wizualizacja wyników
                st.subheader("🔎 Wynik predykcji")
                st.dataframe(converted_df)
                
                # Statystyki predykcji
                st.metric("Liczba przewidzianych awarii", sum(y_pred))
                
                # 📥 Pobranie wyniku jako CSV
                csv = converted_df.to_csv(index=False, sep=';').encode('utf-8')
                st.download_button(
                    "⬇️ Pobierz wynik jako CSV", 
                    data=csv, 
                    file_name=f"predykcja_awarii_{extract_date_from_filename(uploaded_file.name)}.csv",
                    mime='text/csv'
                )
                
            except Exception as e:
                st.error(f"Błąd podczas predykcji: {str(e)}")
