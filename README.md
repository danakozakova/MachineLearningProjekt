# Predikcia dažďa v Austrálii 🌧️

Strojové učenie na predikciu zrážok z denných meteorologických pozorovaní – **klasifikácia** (bude zajtra pršať?) aj **regresia** (koľko naprší). Projekt pokrýva celý pipeline od exploračnej analýzy cez imputáciu chýbajúcich dát a feature engineering až po tréning a porovnanie modelov. [REPORT.md](REPORT_rain_prediction_australia_project.md)

> 🇬🇧 *End-to-end ML project on Australian daily weather data: predicting next-day rain (classification) and rainfall amount (regression). Covers EDA, region-aware missing-data imputation, feature engineering, and model comparison with time-series cross-validation. Write-up is in Slovak — see [REPORT.md](REPORT_rain_prediction_australia_project.md).*

## Výsledky v skratke

**Klasifikácia `RainTomorrow`** (dataset nevyvážený ~78/22, preto ROC-AUC a F1 namiesto samotnej accuracy):

| Model | Accuracy | ROC-AUC | F1 |
|:--|--:|--:|--:|
| Random Forest | 0.85 | **0.87** | 0.59 |
| Logistic Regression | 0.84 | 0.87 | 0.59 |
| Decision Tree | 0.78 | 0.69 | 0.52 |

Recall na dažďových dňoch (~0.45) ukazuje typický kompromis pri nevyváženej klasifikácii – model uprednostňuje presnosť 
pred záchytom menšinovej triedy. Riešiteľné cez `class_weight='balanced'` alebo resampling (napr. SMOTE), za cenu nižšej precision.

**Regresia `RISK_MM`** (na `log(RISK_MM + 1)`, časové rozdelenie dát):

| Model | R² | MAE |
|:--|--:|--:|
| Gradient Boosting | **0.85** | 0.40 |
| Random Forest | 0.83 | 0.44 |
| Linear Regression | 0.81 | 0.49 |

## Čo projekt obsahuje

- **Exploračná analýza** – chýbajúce hodnoty (mechanizmus MAR závislý na lokalite), nevyváženosť tried, sezónne vzory, vzťah smeru vetra a dažďa, detekcia a oprava chybných meraní (tlak, oblačnosť).
- **Imputácia s ohľadom na región** – stanice klastrované K-Means do 9 regiónov podľa súradníc, imputácia v rámci regiónu + binárne indikátory chýbania (`*_imputed`).
- **Feature engineering** – cyklické sin/cos kódovanie smerov vetra, zmeny veličín počas dňa, logaritmické transformácie skosených premenných, časové príznaky.
- **Modelovanie** – 4 feature sety × viac modelov, poctivá **TimeSeriesSplit** validácia (dáta majú časovú štruktúru, náhodné rozdelenie by spôsobilo leakage).

Podrobné zhrnutie s grafmi je v **[REPORT.md](REPORT.md)**.

## Dáta

Denné meteorologické pozorovania z ~49 lokalít v Austrálii, roky 2008–2017 (~156 000 záznamov). Dáta vlastní **Australian Bureau of Meteorology (BOM)**, http://www.bom.gov.au/climate. Ide o verejne dostupný dataset (v ML komunite známy ako *Rain in Australia* / *weatherAUS*).

Zdrojový súbor `data/australia_weather.xlsx` je v repe. Odvodené súbory (imputovaný a transformovaný dataset) sa negenerujú do repa – vytvorí ich notebook pri spustení.

## Spustenie

```bash
git clone https://github.com/danakozakova/MachineLearningProjekt.git
cd MachineLearningProjekt
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl
jupyter lab rain_prediction_australia.ipynb
```

> 💡 Notebook obsahuje grafy, takže výsledky vidno aj bez spustenia. Ak by ho GitHub nevykreslil, otvor ho cez [nbviewer](https://nbviewer.org/github/danakozakova/MachineLearningProjekt/blob/main/rain_prediction_australia.ipynb).

## Kontext

Semestrálny projekt v predmete IB031 (Úvod do strojového učenia) na Fakulte informatiky na Masarykovej univerzite v Brne. Celý projekt – exploračná analýza, imputácia, feature engineering aj modelovanie – je moja samostatná práca. Projekt bol ešte doplnený trénovaním modelov od spolužiakov.
