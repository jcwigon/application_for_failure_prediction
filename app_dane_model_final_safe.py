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
    # Pobierz listę stacji, których oczekuje model (jeśli dostępna)
    if hasattr(model, 'feature_names_in_'):
        expected_stations = set(model.feature_names_in_)
except Exception as e:
    st.error(f"Błąd podczas wczytywania modelu: {str(e)}")
    st.stop()

def process_uploaded_file(uploaded_file):
    try:
        # Wczytaj CSV
        df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8')
        
        # Sprawdź wymagane kolumny (case insensitive)
        df.columns = df.columns.str.lower()
        required_cols = ['machinecode', 'linecode']
        
        if not all(col in df.columns for col in required_cols):
            st.error("Brak wymaganych kolumn 'machinecode' lub 'linecode' w pliku")
            return None
            
        # Wyczyść dane
        df['Stacja'] = df['machinecode'].str.extract(r'([A-Za-z0-9]+)')[0]
        df['Linia'] = df['linecode'].str.extract(r'([A-Za-z0-9]+)')[0]
        
        # Data jutra (z nazwy pliku lub dzisiejsza +1 dzień)
        date_match = re.search(r'DispatchHistory--(\d{4}-\d{2}-\d{2})', uploaded_file.name)
        data_dzienna = pd.to_datetime(date_match.group(1)) if date_match else pd.Timestamp.now() + pd.Timedelta(days=1)
        
        # Przygotuj finalny DataFrame
        result = df[['Stacja', 'Linia']].drop_duplicates()
        result['data_dzienna'] = data_dzienna
        
        return result
    
    except Exception as e:
        st.error(f"Błąd przetwarzania pliku: {str(e)}")
        return None

def prepare_features(df):
    # Kodowanie cech zgodnie z wymaganiami modelu
    X = pd.get_dummies(df['Stacja'])
    
    # Uzupełnij brakujące kolumny
    if hasattr(model, 'feature_names_in_'):
        missing_cols = set(model.feature_names_in_) - set(X.columns)
        for col in missing_cols:
            X[col] = 0
        X = X[model.feature_names_in_]
    
    return X

# UI do wgrywania plików
uploaded_file = st.file_uploader("📤 Wgraj plik DispatchHistory--*.csv", type=['csv'])

if uploaded_file:
    with st.spinner("Przetwarzanie danych..."):
        df = process_uploaded_file(uploaded_file)
        
        if df is not None:
            # Przygotuj dane do predykcji
            X = prepare_features(df)
            
            # Wykonaj predykcję
            df['Predykcja awarii'] = model.predict(X)
            df['Predykcja awarii'] = df['Predykcja awarii'].map({0: "🟢 Brak", 1: "🔴 Będzie"})
            
            # Pokaż wszystkie dostępne linie
            linie = sorted(df['Linia'].unique())
            if len(linie) == 0:
                st.error("Nie znaleziono żadnych linii w danych!")
                st.stop()
                
            wybrana_linia = st.selectbox("🏭 Wybierz linię", linie)
            
            # Filtruj i wyświetl wyniki
            df_filtered = df[df['Linia'] == wybrana_linia].copy()
            df_filtered = df_filtered.drop_duplicates(subset=['Stacja'])
            df_filtered.insert(0, "Lp.", range(1, len(df_filtered)+1))
            
            # Statystyki
            liczba_awarii = (df_filtered['Predykcja awarii'] == '🔴 Będzie').sum()
            st.metric("🔧 Przewidywane awarie", f"{liczba_awarii} stacji")
            
            # Tabela wyników
            st.dataframe(
                df_filtered[['Lp.', 'Linia', 'Stacja', 'Predykcja awarii']],
                use_container_width=True
            )
            
            # Przyciski eksportu
            csv = df_filtered.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Pobierz CSV", data=csv, file_name="predykcja.csv", mime="text/csv")
else:
    st.warning("Proszę wgrać plik z danymi")
