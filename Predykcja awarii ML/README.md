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
- `app.py` – główny plik aplikacji Streamlit
- `model_predykcji_awarii_lightgbm.pkl` – wytrenowany model LightGBM
- `dane_predykcja_3dni.csv` – dane wejściowe do testów
- `requirements.txt` – biblioteki wymagane do uruchomienia

## ▶️ Jak uruchomić lokalnie?

```bash
git clone https://github.com/twoj-login/predykcja-awarii-ml.git
cd predykcja-awarii-ml
pip install -r requirements.txt
streamlit run app.py
```

## 🌐 Wersja online (Streamlit Cloud)

Aplikację możesz uruchomić bez instalacji [TU WSTAW LINK DO STREAMLITA]

---

## 👨‍💻 Autor
Jakub Ćwigoń – Inżynier produkcji i Data Science 🚀