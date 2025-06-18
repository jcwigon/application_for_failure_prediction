import streamlit as st
import pandas as pd
import joblib
import re
from io import BytesIO

st.set_page_config(page_title="Predykcja awarii", page_icon="🛠", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .uploadedFile {
        padding: 12px;
        background: #f0f2f6;
        border-radius: 5px;
        margin: 10px 0;
    }
    .errorBox {
        padding: 15px;
        background: #ffebee;
        border-left: 4px solid #f44336;
        margin: 15px 0;
    }
    .successBox {
        padding: 15px;
        background: #e8f5e9;
        border-left: 4px solid #4caf50;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛠 Predykcja awarii – 1 dzień do przodu")
st.info("Aplikacja przewiduje, czy jutro wystąpi awaria na stacji.")

# Wczytaj model
try:
    model = joblib.load("model_predykcji_awarii_lightgbm.pkl")
    if hasattr(model, 'feature_names_in_'):
        expected_stations = set(model.feature_names_in_)
except Exception as e:
    st.error(f"Błąd podczas wczytywania modelu: {str(e)}")
    st.stop()

def clean_station_name(name):
    return re.sub(r'[^A-Z0-9]', '', str(name).upper()) if pd.notna(name) else None

def clean_line_name(name):
    match = re.search(r'([A-Z]{2,4}\d{0,3})', str(name).upper())
    return match.group(1) if match else None

# UI - Wybór źródła danych
st.markdown("## Wybierz źródło danych:")
data_source = st.radio("", ["Domyślne dane", "Wgraj plik DispatchHistory"],
                      horizontal=True, label_visibility="collapsed")

if data_source == "Domyślne dane":
    try:
        df = pd.read_csv("dane_predykcja_1dzien.csv")
        df['data_dzienna'] = pd.to_datetime(df['data_dzienna'])
        df = df[df['data_dzienna'] == df['data_dzienna'].max()]
        
        df['Linia'] = df['Linia'].apply(clean_line_name)
        df = df.dropna(subset=['Linia'])
        
        st.markdown(f"**Dzień:** Jutro")
        
        linie = sorted(df['Linia'].dropna().unique())
        if not linie:
            st.error("Brak poprawnych linii w danych domyślnych!")
            st.stop()
            
        wybrana_linia = st.selectbox("🏭 Wybierz linię", linie)
        
        X = df[['Stacja']].copy()
        X['Stacja'] = X['Stacja'].astype(str)
        X_encoded = pd.get_dummies(X, drop_first=False)
        
        if hasattr(model, 'feature_names_in_'):
            missing_cols = set(model.feature_names_in_) - set(X_encoded.columns)
            for col in missing_cols:
                X_encoded[col] = 0
            X_encoded = X_encoded[model.feature_names_in_]
        
        df['Predykcja awarii'] = model.predict(X_encoded)
        df['Predykcja awarii'] = df['Predykcja awarii'].map({0: "🟢 Brak", 1: "🔴 Będzie"})
        
        df_filtered = df[df['Linia'] == wybrana_linia].copy()
        df_filtered = df_filtered.drop_duplicates(subset=['Stacja'])
        df_filtered.insert(0, "Lp.", range(1, len(df_filtered)+1))
        
    except Exception as e:
        st.error(f"Błąd przetwarzania domyślnych danych: {str(e)}")
        st.stop()
else:
    st.markdown("## Prześlij plik DispatchHistory")
    
    uploaded_file = st.file_uploader(
        "Wybierz plik CSV",
        type=["csv"],
        accept_multiple_files=False,
        help="Plik powinien zawierać kolumny 'machinecode' i 'linecode'"
    )
    
    if uploaded_file is not None:
        st.markdown(f"""
        <div class="uploadedFile">
            <strong>Wybrany plik:</strong> {uploaded_file.name}
        </div>
        """, unsafe_allow_html=True)
        
        try:
            content = uploaded_file.getvalue().decode('utf-8-sig')
            df = pd.read_csv(BytesIO(content.encode('utf-8')))
            
            # Walidacja kolumn
            required_cols = {'machinecode', 'linecode'}
            if not required_cols.issubset(set(df.columns.str.lower())):
                missing = required_cols - set(df.columns.str.lower())
                st.markdown(f"""
                <div class="errorBox">
                    <strong>Błąd:</strong> Brak wymaganych kolumn: {', '.join(missing)}
                </div>
                """, unsafe_allow_html=True)
                st.stop()
            
            # Przetwarzanie danych
            df['Stacja'] = df['machinecode'].apply(clean_station_name)
            df['Linia'] = df['linecode'].apply(clean_line_name)
            df = df.dropna(subset=['Stacja', 'Linia'])
            
            if df.empty:
                st.markdown("""
                <div class="errorBox">
                    <strong>Błąd:</strong> Brak poprawnych danych w pliku
                </div>
                """, unsafe_allow_html=True)
                st.stop()
            
            # Data z nazwy pliku
            date_match = re.search(r'DispatchHistory[-–](\d{4}-\d{2}-\d{2})', uploaded_file.name)
            data_dzienna = pd.to_datetime(date_match.group(1)) if date_match else pd.Timestamp.now() + pd.Timedelta(days=1)
            
            # Przygotowanie danych dla modelu
            stations_with_failure = set(df['Stacja'].unique())
            all_stations = expected_stations if hasattr(model, 'feature_names_in_') else stations_with_failure
            
            result = []
            for station in all_stations:
                line = df[df['Stacja'] == station]['Linia'].iloc[0] if station in df['Stacja'].values else None
                if line:
                    result.append({
                        'Stacja': station,
                        'Linia': line,
                        'data_dzienna': data_dzienna,
                        'czy_wystapila_awaria': 1 if station in stations_with_failure else 0
                    })
            
            df_processed = pd.DataFrame(result) if result else None
            
            if df_processed is None:
                st.markdown("""
                <div class="errorBox">
                    <strong>Błąd:</strong> Nie udało się przetworzyć danych
                </div>
                """, unsafe_allow_html=True)
                st.stop()
            
            st.markdown(f"""
            <div class="successBox">
                <strong>Dzień:</strong> Jutro ({data_dzienna.strftime('%Y-%m-%d')})
            </div>
            """, unsafe_allow_html=True)
            
            linie = sorted(df_processed['Linia'].dropna().unique())
            if not linie:
                st.error("Nie znaleziono poprawnych linii w danych!")
                st.stop()
                
            wybrana_linia = st.selectbox("🏭 Wybierz linię", linie)
            
            X = df_processed[['Stacja']].copy()
            X['Stacja'] = X['Stacja'].astype(str)
            X_encoded = pd.get_dummies(X['Stacja'])
            
            if hasattr(model, 'feature_names_in_'):
                missing_cols = set(model.feature_names_in_) - set(X_encoded.columns)
                for col in missing_cols:
                    X_encoded[col] = 0
                X_encoded = X_encoded[model.feature_names_in_]
            
            df_processed['Predykcja awarii'] = model.predict(X_encoded)
            df_processed['Predykcja awarii'] = df_processed['Predykcja awarii'].map({0: "🟢 Brak", 1: "🔴 Będzie"})
            
            df_filtered = df_processed[df_processed['Linia'] == wybrana_linia].copy()
            df_filtered = df_filtered.drop_duplicates(subset=['Stacja'])
            df_filtered.insert(0, "Lp.", range(1, len(df_filtered)+1))
            
        except Exception as e:
            st.markdown(f"""
            <div class="errorBox">
                <strong>Błąd przetwarzania:</strong> {str(e)}
            </div>
            """, unsafe_allow_html=True)
            st.stop()
    else:
        st.info("Proszę wybrać plik CSV do analizy")
        st.stop()

# Wyświetl wyniki
if 'df_filtered' in locals():
    st.divider()
    liczba_awarii = (df_filtered['Predykcja awarii'] == '🔴 Będzie').sum()
    st.metric(label="🔧 Przewidywane awarie", value=f"{liczba_awarii} stacji")
    
    st.dataframe(
        df_filtered[['Lp.', 'Linia', 'Stacja', 'Predykcja awarii']],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Lp.": st.column_config.NumberColumn(width="small"),
            "Linia": st.column_config.TextColumn(width="medium"),
            "Stacja": st.column_config.TextColumn(width="large"),
            "Predykcja awarii": st.column_config.TextColumn(width="medium")
        }
    )
    
    col1, col2 = st.columns(2)
    with col1:
        csv = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Pobierz dane do CSV",
            data=csv,
            file_name="predykcja_awarii.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col2:
        excel_data = BytesIO()
        with pd.ExcelWriter(excel_data, engine='xlsxwriter') as writer:
            df_filtered.to_excel(writer, index=False, sheet_name="Predykcja")
        st.download_button(
            label="⬇️ Pobierz dane do Excel",
            data=excel_data.getvalue(),
            file_name="predykcja_awarii.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
