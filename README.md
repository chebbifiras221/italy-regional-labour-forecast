# 🇮🇹 Italy Regional Labour Forecast (NUTS2)

This project builds an ML pipeline to predict **next-year unemployment rate** for **Italian NUTS2 regions** using Eurostat regional datasets.

## Data sources (Eurostat)
- Unemployment rate by NUTS2 region: `tgs00010`
- Regional GDP at current market prices: `nama_10r_2gdp`

API is the Eurostat Statistics API (JSON-stat).

## Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt