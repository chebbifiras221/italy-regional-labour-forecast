# 🇮🇹 Italy Regional Labour Market Forecast (NUTS2)

## 📌 Project Overview

This project builds a full **machine learning forecasting pipeline** to
predict **next-year unemployment rates** for Italian **NUTS2 regions**,
using official **Eurostat regional datasets**.

It combines: - Automated data collection via Eurostat API (JSON-stat) -
Panel dataset construction (region × year) - Feature engineering (lags +
GDP growth) - Supervised learning models (Ridge, Random Forest) -
Interactive Streamlit dashboard (visual analytics + forecasts)

------------------------------------------------------------------------

## 🗂 Project Structure

italy-regional-labour-forecast/
│
├── app/ # Streamlit dashboard
│ └── dashboard.py
│
├── src/ # Core data & ML pipeline
│ ├── build_dataset.py
│ ├── eurostat_api.py
│ ├── features.py
│ ├── train_models.py
│ ├── clustering.py
│ └── utils.py
│
├── data/ # Data storage
│ ├── raw/
│ ├── processed/
│ └── geo/
│
├── models/ # Saved trained models
│
├── requirements.txt # Dependencies
└── run_pipeline.py # Main pipeline runner

------------------------------------------------------------------------

## 📊 Data Sources

All data is pulled directly from the **Eurostat Statistics API
(JSON-stat 2.0)**.

Datasets used:

-   Unemployment rate by NUTS2: `tgs00010`
-   Regional GDP by NUTS2: `nama_10r_2gdp`

------------------------------------------------------------------------

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository

    git clone https://github.com/chebbifiras221/italy-regional-labour-forecast.git
    cd italy-regional-labour-forecast

### 2️⃣ Create a virtual environment

**Windows**

    python -m venv .venv
    .venv\Scripts\activate

**Mac/Linux**

    python -m venv .venv
    source .venv/bin/activate

### 3️⃣ Install dependencies

    pip install --upgrade pip
    pip install -r requirements.txt

------------------------------------------------------------------------

## ▶️ Run the Full Pipeline

    python run_pipeline.py

------------------------------------------------------------------------

## 📈 Launch the Dashboard

    streamlit run app/dashboard.py

------------------------------------------------------------------------

## 🧠 Modeling Approach

Target variable: Next-year unemployment rate (shifted by -1 per region)

Features: - Current unemployment rate - Lag-1 unemployment rate -
Current GDP - Lag-1 GDP - GDP year-over-year growth - Region one-hot
encoding

Models: - Ridge Regression - Random Forest

Evaluation metric: - RMSE

------------------------------------------------------------------------

## 🗺 Optional GeoJSON

For choropleth maps, place:

data/geo/italy_nuts2.geojson

------------------------------------------------------------------------

## 🚀 Skills Demonstrated

-   API data extraction
-   Panel data engineering
-   Time-based forecasting
-   Supervised ML
-   Interactive dashboarding

------------------------------------------------------------------------
## 📸 Screenshots

### Dashboard / Maps / Charts

![Screenshot 1](./visuals/ss1.png)  
![Screenshot 2](./visuals/ss2.png)  
![Screenshot 3](./visuals/ss3.png)  
![Screenshot 4](./visuals/ss4.png)  
![Screenshot 5](./visuals/ss5.png)  

------------------------------------------------------------------------

## 📜 License

No license file currently included.
