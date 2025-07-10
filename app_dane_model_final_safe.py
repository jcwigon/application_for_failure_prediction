import streamlit as st
import pandas as pd
import joblib
import re
from io import BytesIO
from datetime import datetime, timedelta

st.set_page_config(page_title="Predykcja awarii", page_icon="🛠", layout="wide")

# ... (CSS i nagłówki bez zmian)

# Wczytaj nowy model (po treningu na dynamicznych cechach!)
try:
    model = joblib.load("model_predykcji_awarii_lightgbm_dynamic.pkl")  # <-- nowa nazwa modelu!
except Exception as e:
    st.error(f"Błąd podczas wczytywania modelu: {str(e)}")
    st.stop()

# Nazwy wymaganych cech
FEATURE_COLS = [
    'awarie_7dni',
    'awarie_30dni',
    'dni_od_ostatniej_awarii',
    'dni_bez_awarii_z_rzedu',
    'czy_wczoraj_byla_awaria'
]

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

        # Sprawdź dynamiczne kolumny
        missing_cols = set(FEATURE_COLS) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Brak wymaganych kolumn z cechami dynamicznymi: {', '.join(missing_cols)}")

        # Stacja, Linia – pomocnicze do filtrowania
        if not {'Stacja', 'Linia'}.issubset(df.columns):
            raise ValueError("Brak wymaganych kolumn: Stacja, Linia")

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
        df = pd.read_csv("dane_predykcja_1dzien_cechy.csv")  # <-- plik z cechami dynamicznymi!
        df['data_dzienna'] = pd.to_datetime(df['data_dzienna'])
        df = df[df['data_dzienna'] == df['data_dzienna'].max()]

        jutro = datetime.now() + timedelta(days=1)
        st.markdown(f"📅 **Prediction for tomorrow:** {jutro.strftime('%d.%m.%Y')}")

        st.markdown("""
        <div class="demo-info">
            ℹ️ <strong>Demo mode</strong><br>
            You are using demo mode. For real predictions, switch to "Upload DispatchHistory file".
        </div>
        """, unsafe_allow_html=True)

        linie = sorted(df['Linia'].dropna().unique())
        if not linie:
            st.error("No valid lines in demo data!")
            st.stop()

        wybrana_linia = st.selectbox("🏭 Select line", linie)

        # Przygotuj dane do predykcji
        df = df.dropna(subset=FEATURE_COLS)
        X = df[FEATURE_COLS]

        df['Predykcja awarii'] = model.predict(X)
        df['Predykcja awarii'] = df['Predykcja awarii'].map({0: "🟢 No Failure", 1: "🔴 Failure"})

        df_filtered = df[df['Linia'] == wybrana_linia].drop_duplicates(subset=['Stacja'])
        df_filtered.insert(0, "Lp.", range(1, len(df_filtered)+1))

    except Exception as e:
        st.error(f"Błąd przetwarzania danych demo: {str(e)}")
        st.stop()
else:
    st.markdown("## Upload DispatchHistory file in .CSV format")

    with st.container():
        st.markdown('<div class="file-upload-box">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drop or select a CSV file here",
            type=["csv"],
            accept_multiple_files=False,
            key="file_uploader",
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        st.markdown(f"""
        <div class="file-info">
            <strong>Selected file:</strong> {uploaded_file.name}
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("Processing file..."):
            df = validate_uploaded_file(uploaded_file)
            if df is None:
                st.stop()

            try:
                if df.empty:
                    raise ValueError("No valid data after processing file")

                jutro = datetime.now() + timedelta(days=1)
                st.markdown(f"📅 **Prediction for tomorrow:** {jutro.strftime('%d.%m.%Y')}")

                linie = sorted(df['Linia'].dropna().unique())
                if not linie:
                    st.error("No valid lines in data!")
                    st.stop()

                wybrana_linia = st.selectbox("🏭 Select line", linie)

                df = df.dropna(subset=FEATURE_COLS)
                X = df[FEATURE_COLS]

                df['Predykcja awarii'] = model.predict(X)
                df['Predykcja awarii'] = df['Predykcja awarii'].map({0: "🟢 No Failure", 1: "🔴 Failure"})

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
    liczba_awarii = (df_filtered['Predykcja awarii'] == '🔴 Failure').sum()
    st.metric(label="🔧 Predicted failures", value=f"{liczba_awarii} stations")

    st.dataframe(
        df_filtered[['Lp.', 'Linia', 'Stacja', 'Predykcja awarii']],
        use_container_width=True
    )

    col1, col2 = st.columns(2)
    with col1:
        csv = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download CSV",
            data=csv,
            file_name="prediction_results.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col2:
        excel_data = BytesIO()
        with pd.ExcelWriter(excel_data, engine='xlsxwriter') as writer:
            df_filtered.to_excel(writer, index=False, sheet_name="Prediction")
        st.download_button(
            label="⬇️ Download Excel",
            data=excel_data.getvalue(),
            file_name="prediction_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

            file_name="predykcja_awarii.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
