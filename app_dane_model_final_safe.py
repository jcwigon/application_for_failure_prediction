import streamlit as st
import pandas as pd
import joblib
import re
from io import BytesIO
from datetime import datetime

st.set_page_config(page_title="Predykcja awarii", page_icon="🛠", layout="wide")

st.title("🛠 Predykcja awarii – 1 dzień do przodu")
st.info("Aplikacja przewiduje, czy jutro wystąpi awaria na stacji.")

# 📦 Wczytaj model
try:
    model = joblib.load("model_predykcji_awarii_lightgbm.pkl")
except Exception as e:
    st.error(f"Błąd podczas wczytywania modelu: {str(e)}")
    st.stop()

def convert_dispatch_to_model_format(uploaded_file):
    """Konwertuje plik DispatchHistory do formatu dane_predykcja_1dzien.csv"""
    try:
        # Wczytaj plik z różnymi separatorami
        for sep in [';', ',', '\t']:
            try:
                df = pd.read_csv(uploaded_file, sep=sep, encoding='utf-8')
                if len(df.columns) > 1:
                    break
            except:
                continue
        else:
            st.error("Nie można odczytać pliku - sprawdź separator (powinien być ; , lub tab)")
            return None

        # Sprawdź wymagane kolumny
        df.columns = df.columns.str.strip().str.lower()
        if 'machinecode' not in df.columns or 'linecode' not in df.columns:
            st.error("Brak wymaganych kolumn 'machinecode' lub 'linecode' w pliku")
            return None

        # Wyczyść i przygotuj dane
        df['Stacja'] = df['machinecode'].astype(str).str.extract(r'([A-Za-z0-9]+)')[0]
        df['Linia'] = df['linecode'].astype(str).str.extract(r'([A-Za-z0-9]+)')[0]
        
        # Data z nazwy pliku lub jutro
        date_match = re.search(r'DispatchHistory--(\d{4}-\d{2}-\d{2})', uploaded_file.name)
        data_dzienna = pd.to_datetime(date_match.group(1)) if date_match else pd.Timestamp.now() + pd.Timedelta(days=1)
        
        # Stwórz pełny zestaw danych (1 dla awarii, 0 dla braku)
        all_stations = set(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else set(df['Stacja'].unique())
        stations_with_failure = set(df['Stacja'].unique())
        
        result = []
        for station in all_stations:
            line = df[df['Stacja'] == station]['Linia'].iloc[0] if station in df['Stacja'].values else station[:4]
            result.append({
                'Stacja': station,
                'Linia': line,
                'data_dzienna': data_dzienna,
                'czy_wystapila_awaria': 1 if station in stations_with_failure else 0
            })
        
        return pd.DataFrame(result)
        
    except Exception as e:
        st.error(f"Błąd przetwarzania pliku: {str(e)}")
        return None

# UI do wyboru źródła danych
data_source = st.radio("Wybierz źródło danych:", ["Domyślne dane", "Wgraj plik DispatchHistory"])

if data_source == "Domyślne dane":
    # Użyj oryginalnego kodu z domyślnymi danymi
    try:
        df = pd.read_csv("dane_predykcja_1dzien.csv")
        df['data_dzienna'] = pd.to_datetime(df['data_dzienna'])
        df = df[df['data_dzienna'] == df['data_dzienna'].max()]
        
        st.markdown(f"**Dzień:** Jutro")
        
        linie = sorted(df['Stacja'].str.extract(r'(^[A-Z]{2,}[0-9]{2,})')[0].dropna().unique())
        wybrana_linia = st.selectbox("🏭 Wybierz linię", linie)
        
        X = df[['Stacja']].copy()
        X['Stacja'] = X['Stacja'].astype(str)
        X_encoded = pd.get_dummies(X, drop_first=False)
        
        # Dopasuj kolumny do wymagań modelu
        if hasattr(model, 'feature_names_in_'):
            missing_cols = set(model.feature_names_in_) - set(X_encoded.columns)
            for col in missing_cols:
                X_encoded[col] = 0
            X_encoded = X_encoded[model.feature_names_in_]
        
        df['Predykcja awarii'] = model.predict(X_encoded)
        df['Predykcja awarii'] = df['Predykcja awarii'].map({0: "🟢 Brak", 1: "🔴 Będzie"})
        
        df_filtered = df[df['Stacja'].str.startswith(wybrana_linia)].copy()
        df_filtered = df_filtered.drop_duplicates(subset=['Stacja'])
        
        if 'Linia' in df_filtered.columns:
            df_filtered.drop(columns=['Linia'], inplace=True)
        df_filtered.insert(1, "Linia", wybrana_linia)
        
    except Exception as e:
        st.error(f"Błąd przetwarzania domyślnych danych: {str(e)}")
        st.stop()
else:
    uploaded_file = st.file_uploader("📤 Wgraj plik DispatchHistory--*.csv", type=['csv'])
    if not uploaded_file:
        st.stop()
        
    with st.spinner("Przetwarzanie danych..."):
        df = convert_dispatch_to_model_format(uploaded_file)
        if df is None:
            st.stop()
            
        st.markdown(f"**Dzień:** Jutro")
        
        linie = sorted(df['Stacja'].str.extract(r'(^[A-Z]{2,}[0-9]{2,})')[0].dropna().unique())
        wybrana_linia = st.selectbox("🏭 Wybierz linię", linie)
        
        X = df[['Stacja']].copy()
        X['Stacja'] = X['Stacja'].astype(str)
        X_encoded = pd.get_dummies(X, drop_first=False)
        
        # Dopasuj kolumny do wymagań modelu
        if hasattr(model, 'feature_names_in_'):
            missing_cols = set(model.feature_names_in_) - set(X_encoded.columns)
            for col in missing_cols:
                X_encoded[col] = 0
            X_encoded = X_encoded[model.feature_names_in_]
        
        df['Predykcja awarii'] = model.predict(X_encoded)
        df['Predykcja awarii'] = df['Predykcja awarii'].map({0: "🟢 Brak", 1: "🔴 Będzie"})
        
        df_filtered = df[df['Stacja'].str.startswith(wybrana_linia)].copy()
        df_filtered = df_filtered.drop_duplicates(subset=['Stacja'])
        
        if 'Linia' in df_filtered.columns:
            df_filtered.drop(columns=['Linia'], inplace=True)
        df_filtered.insert(1, "Linia", wybrana_linia)

# Wyświetl wyniki
if 'df_filtered' in locals():
    df_filtered.insert(0, "Lp.", range(1, len(df_filtered) + 1))
    
    liczba_awarii = (df_filtered['Predykcja awarii'] == '🔴 Będzie').sum()
    st.metric(label="🔧 Przewidywane awarie", value=f"{liczba_awarii} stacji")
    
    st.dataframe(
        df_filtered[['Lp.', 'Linia', 'Stacja', 'Predykcja awarii']].reset_index(drop=True),
        use_container_width=True
    )
    
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Pobierz dane do CSV",
        data=csv,
        file_name="predykcja_1dzien.csv",
        mime="text/csv"
    )
    
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
