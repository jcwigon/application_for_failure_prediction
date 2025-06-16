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

def convert_dispatch_to_model_format(uploaded_file):
    """Konwertuje wgrywany plik DispatchHistory do formatu dane_predykcja_1dzien.csv"""
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

        # Sprawdź wymagane kolumny (case insensitive)
        df.columns = df.columns.str.lower()
        if 'machinecode' not in df.columns or 'linecode' not in df.columns:
            st.error("Brak wymaganych kolumn 'machinecode' lub 'linecode' w pliku")
            return None

        # Wyczyść i przygotuj dane
        df['Stacja'] = df['machinecode'].astype(str).str.extract(r'([A-Za-z0-9]+)')[0]
        df['Linia'] = df['linecode'].astype(str).str.extract(r'([A-Za-z0-9]+)')[0]
        
        # Data z nazwy pliku lub dzisiaj +1 dzień
        date_match = re.search(r'DispatchHistory--(\d{4}-\d{2}-\d{2})', uploaded_file.name)
        data_dzienna = pd.to_datetime(date_match.group(1)) if date_match else pd.Timestamp.now() + pd.Timedelta(days=1)
        
        # Stwórz finalny DataFrame w odpowiednim formacie
        result = df[['Stacja', 'Linia']].drop_duplicates()
        result['data_dzienna'] = data_dzienna
        result['Predykcja awarii'] = None  # Tymczasowo puste
        
        return result
    
    except Exception as e:
        st.error(f"Błąd przetwarzania pliku: {str(e)}")
        return None

def make_predictions(df):
    """Wykonuje predykcję na danych"""
    try:
        # Przygotuj dane do predykcji
        X = pd.get_dummies(df['Stacja'])
        
        # Dopasuj do wymagań modelu
        if hasattr(model, 'feature_names_in_'):
            missing = set(model.feature_names_in_) - set(X.columns)
            for col in missing:
                X[col] = 0
            X = X[model.feature_names_in_]
        
        # Wykonaj predykcję
        df['Predykcja awarii'] = model.predict(X)
        df['Predykcja awarii'] = df['Predykcja awarii'].map({0: "🟢 Brak", 1: "🔴 Będzie"})
        
        return df
    except Exception as e:
        st.error(f"Błąd predykcji: {str(e)}")
        return None

# UI do wyboru źródła danych
data_source = st.radio("Wybierz źródło danych:", ["Domyślne dane", "Wgraj plik DispatchHistory"])

if data_source == "Domyślne dane":
    # Użyj domyślnych danych
    try:
        df = pd.read_csv("dane_predykcja_1dzien.csv")
        df['data_dzienna'] = pd.to_datetime(df['data_dzienna'])
        
        # Użyj najnowszej daty z danych
        data_jutra = df['data_dzienna'].max()
        df = df[df['data_dzienna'] == data_jutra]
        
        # Wykonaj predykcję
        df_pred = make_predictions(df)
        
    except Exception as e:
        st.error(f"Błąd wczytywania domyślnych danych: {str(e)}")
        st.stop()
else:
    # Wgraj plik DispatchHistory
    uploaded_file = st.file_uploader("📤 Wgraj plik DispatchHistory--*.csv", type=['csv'])
    
    if not uploaded_file:
        st.stop()
        
    with st.spinner("Przetwarzanie danych..."):
        df_pred = convert_dispatch_to_model_format(uploaded_file)
        
        if df_pred is not None:
            df_pred = make_predictions(df_pred)
        else:
            st.stop()

# Jeśli mamy dane, wyświetl interfejs
if df_pred is not None:
    # ⏳ Dzień jutro
    st.markdown(f"**Dzień:** Jutro ({df_pred['data_dzienna'].iloc[0].strftime('%Y-%m-%d')})")
    
    # 📍 Filtr linii
    linie = sorted(df_pred['Linia'].dropna().unique())
    if not linie:
        st.error("Nie znaleziono żadnych linii w danych!")
        st.stop()
        
    wybrana_linia = st.selectbox("🏭 Wybierz linię", linie)
    
    # 🔍 Filtrowanie dla wybranej linii
    df_filtered = df_pred[df_pred['Linia'] == wybrana_linia].copy()
    df_filtered = df_filtered.drop_duplicates(subset=['Stacja'])
    
    # 🧾 Dodaj kolumny i numerację
    df_filtered.insert(0, "Lp.", range(1, len(df_filtered)+1))
    
    # 📋 Wyświetl metrykę
    liczba_awarii = (df_filtered['Predykcja awarii'] == '🔴 Będzie').sum()
    st.metric(label="🔧 Przewidywane awarie", value=f"{liczba_awarii} stacji")
    
    # 📊 Tabela wyników
    st.dataframe(
        df_filtered[['Lp.', 'Linia', 'Stacja', 'Predykcja awarii']],
        use_container_width=True
    )
    
    # 💾 Eksport danych
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Pobierz dane do CSV",
        data=csv,
        file_name="predykcja_awarii.csv",
        mime="text/csv"
    )
    
    # Eksport do Excel
    def to_excel(df):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name="Predykcja")
        return output.getvalue()
    
    st.download_button(
        label="⬇️ Pobierz dane do Excel",
        data=to_excel(df_filtered),
        file_name="predykcja_awarii.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
