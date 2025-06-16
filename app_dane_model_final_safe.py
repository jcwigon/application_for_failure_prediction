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
    # Pobierz listę stacji, których oczekuje model
    if hasattr(model, 'feature_names_in_'):
        expected_features = set(model.feature_names_in_)
except Exception as e:
    st.error(f"Błąd podczas wczytywania modelu: {str(e)}")
    st.stop()

def load_default_data():
    """Wczytuje domyślne dane i wykonuje predykcję"""
    try:
        df = pd.read_csv("dane_predykcja_1dzien.csv")
        df['data_dzienna'] = pd.to_datetime(df['data_dzienna'])
        return df
    except Exception as e:
        st.error(f"Błąd wczytywania domyślnych danych: {str(e)}")
        return None

def convert_uploaded_file(uploaded_file):
    """Konwertuje wgrywany plik DispatchHistory do odpowiedniego formatu"""
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

        # Wyczyść dane
        df['Stacja'] = df['machinecode'].astype(str).str.extract(r'([A-Za-z0-9]+)')[0]
        df['Linia'] = df['linecode'].astype(str).str.extract(r'([A-Za-z0-9]+)')[0]
        
        # Data z nazwy pliku lub jutro
        date_match = re.search(r'DispatchHistory--(\d{4}-\d{2}-\d{2})', uploaded_file.name)
        data_dzienna = pd.to_datetime(date_match.group(1)) if date_match else pd.Timestamp.now() + pd.Timedelta(days=1)
        
        # Stwórz finalny DataFrame
        result = df[['Stacja', 'Linia']].drop_duplicates()
        result['data_dzienna'] = data_dzienna
        
        return result
    except Exception as e:
        st.error(f"Błąd przetwarzania pliku: {str(e)}")
        return None

def prepare_features(df):
    """Przygotowuje dane do predykcji"""
    try:
        # Kodowanie cech
        X = pd.get_dummies(df['Stacja'])
        
        # Uzupełnij brakujące kolumny
        if hasattr(model, 'feature_names_in_'):
            missing = set(model.feature_names_in_) - set(X.columns)
            for col in missing:
                X[col] = 0
            X = X[model.feature_names_in_]
        
        return X
    except Exception as e:
        st.error(f"Błąd przygotowania danych: {str(e)}")
        return None

def make_predictions(df):
    """Wykonuje predykcję i zwraca wyniki"""
    try:
        X = prepare_features(df)
        if X is None:
            return None
            
        df['Predykcja awarii'] = model.predict(X)
        df['Predykcja awarii'] = df['Predykcja awarii'].map({0: "🟢 Brak", 1: "🔴 Będzie"})
        return df
    except Exception as e:
        st.error(f"Błąd predykcji: {str(e)}")
        return None

# UI do wyboru źródła danych
data_source = st.radio("Wybierz źródło danych:", ["Domyślne dane", "Wgraj plik DispatchHistory"])

if data_source == "Domyślne dane":
    df = load_default_data()
    if df is None:
        st.stop()
        
    # Użyj najnowszej daty z danych
    df = df[df['data_dzienna'] == df['data_dzienna'].max()]
    
    # Jeśli domyślne dane już mają predykcje, nie wykonuj ponownie
    if 'Predykcja awarii' not in df.columns:
        df = make_predictions(df)
else:
    uploaded_file = st.file_uploader("📤 Wgraj plik DispatchHistory--*.csv", type=['csv'])
    if not uploaded_file:
        st.stop()
        
    df = convert_uploaded_file(uploaded_file)
    if df is not None:
        df = make_predictions(df)

# Jeśli mamy dane, wyświetl interfejs
if df is not None:
    # ⏳ Dzień jutro
    st.markdown(f"**Dzień:** Jutro ({df['data_dzienna'].iloc[0].strftime('%Y-%m-%d')})")
    
    # 📍 Filtr linii - pokaż wszystkie dostępne linie
    linie = sorted(df['Linia'].dropna().unique())
    if not linie:
        st.error("Nie znaleziono żadnych linii w danych!")
        st.stop()
        
    wybrana_linia = st.selectbox("🏭 Wybierz linię", linie)
    
    # 🔍 Filtrowanie dla wybranej linii
    df_filtered = df[df['Linia'] == wybrana_linia].copy()
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
