# 🔧 Predykcja awarii maszyn – ML (LightGBM + Streamlit)

Projekt służy do przewidywania awarii maszyn na podstawie danych z linii produkcyjnych.  
Został stworzony w Pythonie z użyciem modeli ML i interfejsu Streamlit.

## 🔍 Co potrafi aplikacja?
- Wczytuje dane z 10 linii produkcyjnych na 3 dni do przodu
- Przewiduje, czy na danej stacji wystąpi awaria
- Pokazuje tabelę z predykcjami oraz podsumowanie
- Umożliwia eksport wyników do CSV i XLSX
- Działa jako aplikacja webowa przez Streamlit Cloud

## 📂 Zawartość repozytorium
- `app_predykcja_profesjonalna.py` – główny plik aplikacji Streamlit
- `model_predykcji_awarii_lightgbm.pkl` – wytrenowany model LightGBM
- `dane_predykcja_3dni.csv` – dane wejściowe do testów
- `requirements.txt` – biblioteki wymagane do uruchomienia

## ▶️ Jak uruchomić lokalnie?

```bash
git clone https://github.com/jcwigon/Predykcja-awarii-ML.git
cd Predykcja-awarii-ML
pip install -r requirements.txt
streamlit run app_predykcja_profesjonalna.py
