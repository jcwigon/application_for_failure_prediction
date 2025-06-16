import streamlit as st
import pandas as pd
import joblib
import re
from io import BytesIO

st.set_page_config(page_title="Predykcja awarii", page_icon="🛠", layout="wide")

st.title("🛠 Predykcja awarii – 1 dzień do przodu")
st.info("Aplikacja przewiduje, czy jutro wystąpi awaria na stacji.")

# 📦 Wczytaj model
try:
    model = joblib.load("model_predykcji_awarii_lightgbm.pkl")
except Exception as e:
    st.error(f"Błąd podczas wczytywania modelu: {str(e)}")
    st.stop()

def process_uploaded_file(uploaded_file):
    try:
        # Wczytaj plik CSV z różnymi separatorami
        for sep in [';', ',', '\t']:
            try:
                df = pd.read_csv(uploaded_file, sep=sep, encoding='utf-8')
                if len(df.columns) > 1:
                    break
            except:
                continue
        else:
            st.error("Nie można odczytać pliku - sprawdź separator")
            return None

        # Debug: pokaż nagłówki
        st.write("Znalezione kolumny:", df.columns.tolist())

        # Normalizuj nazwy kolumn
        df.columns = df.columns.str.strip().str.lower()

        # Sprawdź wymagane kolumny
        required = {'machinecode', 'linecode'}
        if not required.issubset(set(df.columns)):
            st.error(f"Brak wymaganych kolumn. Potrzebne: {required}, Znalezione: {set(df.columns)}")
            return None

        # Wyczyść i przygotuj dane
        df['Stacja'] = df['machinecode'].astype(str).str.extract(r'([A-Za-z0-9]+)')[0]
        df['Linia'] = df['linecode'].astype(str).str.extract(r'([A-Za-z0-9]+)')[0]

        # Data z nazwy pliku lub dzisiaj +1 dzień
        date_match = re.search(r'DispatchHistory--(\d{4}-\d{2}-\d{2})', uploaded_file.name)
        df['data_dzienna'] = pd.to_datetime(date_match.group(1)) if date_match else pd.Timestamp.now() + pd.Timedelta(days=1)

        return df[['Stacja', 'Linia', 'data_dzienna']].drop_duplicates()

    except Exception as e:
        st.error(f"Błąd przetwarzania pliku: {str(e)}")
        return None

def make_predictions(df):
    try:
        # Przygotuj dane do predykcji
        X = pd.get_dummies(df['Stacja'])
        
        # Dopasuj do wymagań modelu
        if hasattr(model, 'feature_names_in_'):
            missing = set(model.feature_names_in_) - set(X.columns)
            for col in missing:
                X[col] = 0
            X = X[model.feature_names_in_]
        
        # Predykcja
        df['Predykcja awarii'] = model.predict(X)
        df['Predykcja awarii'] = df['Predykcja awarii'].map({0: "🟢 Brak", 1: "🔴 Będzie"})
        return df
    except Exception as e:
        st.error(f"Błąd predykcji: {str(e)}")
        return None

# Interfejs
uploaded_file = st.file_uploader("📤 Wgraj plik DispatchHistory--*.csv", type=['csv'])

if uploaded_file is not None:
    with st.spinner("Przetwarzanie danych..."):
        df = process_uploaded_file(uploaded_file)
        
        if df is not None:
            df_pred = make_predictions(df)
            
            if df_pred is not None:
                # Wybór linii
                linie = sorted(df_pred['Linia'].dropna().unique())
                if len(linie) == 0:
                    st.error("Nie znaleziono linii w danych!")
                else:
                    wybrana_linia = st.selectbox("🏭 Wybierz linię", linie)
                    
                    # Filtruj wyniki
                    results = df_pred[df_pred['Linia'] == wybrana_linia].copy()
                    results.insert(0, "Lp.", range(1, len(results)+1))
                    
                    # Wyświetl wyniki
                    st.metric("🔧 Przewidywane awarie", 
                             f"{(results['Predykcja awarii'] == '🔴 Będzie').sum()} stacji")
                    
                    st.dataframe(
                        results[['Lp.', 'Linia', 'Stacja', 'Predykcja awarii']],
                        use_container_width=True
                    )
                    
                    # Eksport
                    csv = results.to_csv(index=False).encode('utf-8')
                    st.download_button("⬇️ Pobierz wyniki", data=csv, 
                                     file_name="predykcja_awarii.csv", mime="text/csv")
