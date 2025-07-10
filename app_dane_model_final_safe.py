import streamlit as st
import pandas as pd
import joblib
from io import BytesIO
from datetime import datetime, timedelta

st.set_page_config(page_title="Predykcja awarii", page_icon="🛠", layout="wide")

try:
    model = joblib.load("model_predykcji_awarii_lightgbm.pkl")
except Exception as e:
    st.error(f"Błąd podczas wczytywania modelu: {str(e)}")
    st.stop()

FEATURE_COLS = [
    'awarie_7dni',
    'awarie_30dni',
    'dni_od_ostatniej_awarii',
    'dni_bez_awarii_z_rzedu',
    'czy_wczoraj_byla_awaria'
]

def add_dynamic_features(df):
    df = df.sort_values(['Stacja', 'data_dzienna'])

    def rolling_awarie_7dni(x):
        return x.rolling(window=7, min_periods=1).sum().shift(1).fillna(0)
    df['awarie_7dni'] = df.groupby('Stacja')['czy_wystapila_awaria'].apply(rolling_awarie_7dni).reset_index(level=0, drop=True)

    def rolling_awarie_30dni(x):
        return x.rolling(window=30, min_periods=1).sum().shift(1).fillna(0)
    df['awarie_30dni'] = df.groupby('Stacja')['czy_wystapila_awaria'].apply(rolling_awarie_30dni).reset_index(level=0, drop=True)

    def days_since_last_failure(group):
        last_date = None
        days = []
        for idx, row in group.iterrows():
            if row['czy_wystapila_awaria'] == 1:
                last_date = row['data_dzienna']
                days.append(0)
            elif last_date is None:
                days.append(None)
            else:
                days.append((row['data_dzienna'] - last_date).days)
        return pd.Series(days, index=group.index)
    df['dni_od_ostatniej_awarii'] = df.groupby('Stacja', group_keys=False).apply(days_since_last_failure)

    def days_without_failure(group):
        count = 0
        days = []
        for value in group:
            if value == 0:
                count += 1
            else:
                count = 0
            days.append(count)
        return days
    df['dni_bez_awarii_z_rzedu'] = df.groupby('Stacja')['czy_wystapila_awaria'].transform(days_without_failure)

    df['czy_wczoraj_byla_awaria'] = df.groupby('Stacja')['czy_wystapila_awaria'].shift(1).fillna(0)

    return df

st.title("🛠 Predykcja awarii – 1 dzień do przodu")
st.info("System prognozuje wystąpienie awarii na stacji z wyprzedzeniem 24-godzinnym")

st.markdown("## Wybierz źródło danych:")
data_source = st.radio("", ["Domyślne dane", "Wgraj plik DispatchHistory"],
                      horizontal=True, label_visibility="collapsed")

if data_source == "Domyślne dane":
    try:
        df = pd.read_csv("dane_predykcja_1dzien.csv")
        df['data_dzienna'] = pd.to_datetime(df['data_dzienna'])
        df = add_dynamic_features(df)
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
            try:
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

                # Mapowanie kolumn i rzutowanie na datę (ważne!)
                df['Linia'] = df['linecode']
                df['Stacja'] = df['machinecode']
                df['data_dzienna'] = pd.to_datetime(df['dispatched'], errors='coerce')
                df = df.dropna(subset=['data_dzienna'])  # Usuwa wiersze bez daty
                df['Linia'] = df['Linia'].astype(str)
                df['Stacja'] = df['Stacja'].astype(str)

                # Konwersja do daty bez godziny (do MultiIndex)
                df['data_dzienna'] = df['data_dzienna'].dt.normalize()

                # RZECZYWISTE pary stacja-linia (każda stacja tylko w swojej linii!)
                all_dates = pd.date_range(df['data_dzienna'].min(), df['data_dzienna'].max())
                stacja_linia = df[['Stacja', 'Linia']].drop_duplicates()

                siatka = pd.MultiIndex.from_product(
                    [all_dates, stacja_linia.itertuples(index=False, name=None)],
                    names=['data_dzienna', 'stacja_linia']
                )
                df_full = pd.DataFrame(index=siatka).reset_index()

                # Rozbij tuple (Stacja, Linia) na dwie kolumny
                df_full[['Stacja', 'Linia']] = pd.DataFrame(df_full['stacja_linia'].tolist(), index=df_full.index)
                df_full = df_full.drop(columns=['stacja_linia'])

                df_events = df.groupby(['data_dzienna', 'Stacja', 'Linia']).size().reset_index(name='awaria_event')
                df_events['awaria_event'] = 1

                # PRZED MERGE oba data_dzienna jako datetime64[ns]!
                df_full['data_dzienna'] = pd.to_datetime(df_full['data_dzienna'])
                df_events['data_dzienna'] = pd.to_datetime(df_events['data_dzienna'])

                df_full = df_full.merge(df_events, on=['data_dzienna', 'Stacja', 'Linia'], how='left')
                df_full['czy_wystapila_awaria'] = df_full['awaria_event'].fillna(0).astype(int)

                df_full = add_dynamic_features(df_full)

                linie = sorted(df_full['Linia'].dropna().unique())
                if not linie:
                    st.error("No valid lines in data!")
                    st.stop()

                wybrana_linia = st.selectbox("🏭 Select line", linie)

                # Predykcja tylko dla najnowszego dnia
                max_date = df_full['data_dzienna'].max()
                df_pred = df_full[df_full['data_dzienna'] == max_date]
                df_pred = df_pred[df_pred['Linia'] == wybrana_linia].drop_duplicates(subset=['Stacja'])
                df_pred = df_pred.dropna(subset=FEATURE_COLS)
                X = df_pred[FEATURE_COLS]

                df_pred['Predykcja awarii'] = model.predict(X)
                df_pred['Predykcja awarii'] = df_pred['Predykcja awarii'].map({0: "🟢 No Failure", 1: "🔴 Failure"})

                df_pred.insert(0, "Lp.", range(1, len(df_pred)+1))

                df_filtered = df_pred

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



