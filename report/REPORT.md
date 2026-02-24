# report/REPORT.md 
# Italy Regional Labour Forecast (NUTS2) — Mini Report

## 1. Motivation
Regional labour market disparities are a key policy topic in Italy. This project builds reproducible ML baselines to predict next-year unemployment at NUTS2 level.

## 2. Territorial units
We use NUTS2 regions (Eurostat classification used for regional policy).

## 3. Data
Eurostat regional datasets:
- Unemployment rate by NUTS2 region (`tgs00010`)
- GDP at current market prices by NUTS2 region (`nama_10r_2gdp`)

## 4. Methods
- Panel dataset by region-year
- Features: lags, GDP YoY growth
- Models: Ridge, Random Forest, optional XGBoost
- Evaluation: time-aware split

## 5. Results
Insert metrics from `models/metrics.json` and discuss.

## 6. Limitations
- GDP is current prices; real GDP could be preferable.
- Missingness varies by region/year.
- No causal claims; predictive project only.

## 7. Conclusion
Summarize best model and how the dashboard supports interpretation.