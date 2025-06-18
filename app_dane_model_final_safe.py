import streamlit as st
import pandas as pd
import joblib
import re
from io import BytesIO
from datetime import datetime, timedelta

st.set_page_config(page_title="Predykcja awarii", page_icon="🛠", layout="wide")

# Custom CSS dla lepszego wyglądu
st.markdown("""
<style>
    .file-upload-box {
        border: 2px dashed #ccc;
        border-radius: 5px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
    }
    .file-info {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .error-box {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 10px;
        margin: 10px 0;
    }
    .demo-info {
        background-color: #e3f2fd;
        padding: 12px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛠 Predykcja awarii – 1 dzień do przodu")
st.info("System prognozuje wystąpienie awarii na stacji z wyprzedzeniem 24-godzinnym")

# Wczytanie modelu
try:
    model = joblib.load("model_predykcji_awarii_lightgbm.pkl")
    if hasattr(model, 'feature_names_in_'):
        expected_stations = set(model.feature_names_in_)
except Exception as e:
    st.error(f"Błąd podczas wczytywania modelu: {str(e)}")
    st.stop()

def validate_uploaded_file(uploaded_file):
    try:
        if not uploaded_file.name.lower().endswith('.csv'):
            raise ValueError("Plik musi mieć rozszerzenie .csv")

        content = uploaded_file.getvalue().decode('utf-8-sig')
        for sep in [',', ';', '\t']:
            try:
                df = pd.read_csv(BytesIO(content.encode('utf-8')), sep=sep, engine='python')
                if len(df.columns) > 1:
                    break
            except:
                continue
        else:
            raise ValueError("Nie można odczytać pliku CSV. Sprawdź separator (przecinek, średnik lub tabulator).")

        df.columns = df.columns.str.strip().str.lower()
        required_cols = {'machinecode', 'linecode'}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Brak wymaganych kolumn: {', '.join(missing)}")

        return df
    except Exception as e:
        st.markdown(f"""
        <div class="error-box">
            <strong>Błąd walidacji pliku:</strong> {str(e)}
        </div>
        """, unsafe_allow_html=True)
        return None

st.markdown("## Wybierz źródło danych:")
data_source = st.radio("", ["Domyślne dane", "Wgraj plik DispatchHistory"],
                      horizontal=True, label_visibility="collapsed")

if data_source == "Domyślne dane":
    try:
        df = pd.read_csv("dane_predykcja_1dzien.csv")
        df['data_dzienna'] = pd.to_datetime(df['data_dzienna'])
        df = df[df['data_dzienna'] == df['data_dzienna'].max()]

        # Wyświetlenie daty
        jutro = datetime.now() + timedelta(days=1)
        st.markdown(f"""
        📅 **Prognoza na jutro:** {jutro.strftime('%d.%m.%Y')}
        """)

        # DODANY KOMUNIKAT O TRYBIE DEMONSTRACYJNYM
        st.markdown("""
        <div class="demo-info">
            ℹ️ <strong>Tryb demonstracyjny</strong><br>
            Używasz trybu demonstracyjnego aplikacji, który symuluje działanie aplikacji w celu predykcji.<br>
            Przełącz się na tryb "Wgraj plik DispatchHistory" i wgraj rzeczywiste dane z systemu Leading2Lean.
        </div>
        """, unsafe_allow_html=True)

        linie = sorted(df['Linia'].dropna().unique())
        if not linie:
            st.error("Brak poprawnych linii w danych domyślnych!")
            st.stop()

        wybrana_linia = st.selectbox("🏭 Wybierz linię", linie)

        X = pd.get_dummies(df[['Stacja']], drop_first=False)

        if hasattr(model, 'feature_names_in_'):
            missing_cols = set(model.feature_names_in_) - set(X.columns)
            for col in missing_cols:
                X[col] = 0
            X = X[model.feature_names_in_]

        df['Predykcja awarii'] = model.predict(X)
        df['Predykcja awarii'] = df['Predykcja awarii'].map({0: "🟢 Brak", 1: "🔴 Będzie"})

        df_filtered = df[df['Linia'] == wybrana_linia].drop_duplicates(subset=['Stacja'])
        df_filtered.insert(0, "Lp.", range(1, len(df_filtered)+1))

    except Exception as e:
        st.error(f"Błąd przetwarzania domyślnych danych: {str(e)}")
        st.stop()
else:
    st.markdown("## Prześlij plik DispatchHistory w formacie .CSV")

    with st.container():
        st.markdown('<div class="file-upload-box">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Przeciągnij i upuść plik CSV tutaj lub kliknij, aby wybrać",
            type=["csv"],
            accept_multiple_files=False,
            key="file_uploader",
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        st.markdown(f"""
        <div class="file-info">
            <strong>Wybrany plik:</strong> {uploaded_file.name}
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("Przetwarzanie pliku..."):
            df = validate_uploaded_file(uploaded_file)
            if df is None:
                st.stop()

            try:
                df['Stacja'] = df['machinecode']
                df['Linia'] = df['linecode']
                df = df.dropna(subset=['Stacja', 'Linia'])

                if df.empty:
                    raise ValueError("Brak poprawnych danych po przetworzeniu pliku")

                jutro = datetime.now() + timedelta(days=1)
                st.markdown(f"""
                📅 **Predykcja na jutro:** {jutro.strftime('%d.%m.%Y')}
                """)

                linie = sorted(df['Linia'].dropna().unique())
                if not linie:
                    st.error("Nie znaleziono poprawnych linii w danych!")
                    st.stop()

                wybrana_linia = st.selectbox("🏭 Wybierz linię", linie)

                X = pd.get_dummies(df[['Stacja']], drop_first=False)

                if hasattr(model, 'feature_names_in_'):
                    missing_cols = set(model.feature_names_in_) - set(X.columns)
                    for col in missing_cols:
                        X[col] = 0
                    X = X[model.feature_names_in_]

                df['Predykcja awarii'] = model.predict(X)
                df['Predykcja awarii'] = df['Predykcja awarii'].map({0: "🟢 Brak", 1: "🔴 Będzie"})

                df_filtered = df[df['Linia'] == wybrana_linia].drop_duplicates(subset=['Stacja'])
                df_filtered.insert(0, "Lp.", range(1, len(df_filtered)+1))

            except Exception as e:
                st.markdown(f"""
                <div class="error-box">
                    <strong>Błąd przetwarzania danych:</strong> {str(e)}
                </div>
                """, unsafe_allow_html=True)
                st.stop()

if 'df_filtered' in locals():
    st.divider()
    liczba_awarii = (df_filtered['Predykcja awarii'] == '🔴 Będzie').sum()
    st.metric(label="🔧 Przewidywane awarie", value=f"{liczba_awarii} stacji")

    st.dataframe(
        df_filtered[['Lp.', 'Linia', 'Stacja', 'Predykcja awarii']],
        use_container_width=True
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
