import sys
from pathlib import Path
import json
import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.clustering import run_clustering

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
st.set_page_config(
    page_title="Italy Regional Labour Forecast",
    layout="wide",
    page_icon="📊"
)

st.title("Italy Regional Labour Forecasting System")
st.markdown("""
Machine learning framework for regional unemployment forecasting  
using Eurostat NUTS2 panel data.
""")
st.divider()

DATA_PATH = ROOT / "data" / "processed" / "regional_panel_features.csv"
PRED_PATH = ROOT / "models" / "predictions.csv"
METRICS_PATH = ROOT / "models" / "metrics.json"
GEO_PATH = ROOT / "data" / "geo" / "italy_nuts2.geojson"

if not DATA_PATH.exists():
    st.error("Run pipeline first: python run_pipeline.py")
    st.stop()

df = pd.read_csv(DATA_PATH)

# ----------------------------------------------------
# DATASET INFO
# ----------------------------------------------------
with st.expander("About the Dataset"):
    st.markdown("""
**Source:** Eurostat  
**Territorial Level:** NUTS2 (major socio-economic regions)  
**Frequency:** Annual  

Variables used:
- Unemployment rate (%)
- GDP (current prices)
- Lag variables
- GDP growth rate

NUTS2 refers to the European regional classification system 
used for statistical and economic analysis.
""")

# ----------------------------------------------------
# TABS
# ----------------------------------------------------
tabs = st.tabs([
    "Data Overview",
    "Model Evaluation",
    "Structural Analysis",
    "Forecast",
    "Methodology",
    "Data Dictionary"
])

# ====================================================
# DATA OVERVIEW
# ====================================================
with tabs[0]:

    region_map = df[["geo", "region"]].drop_duplicates().sort_values("region")

    col1, col2 = st.columns([1, 3])

    with col1:
        region_name = st.selectbox(
            "Select Region",
            region_map["region"],
            key="region_tab1"
        )
        region = region_map[region_map["region"] == region_name]["geo"].values[0]
        st.caption(f"Region Code: {region}")

    region_df = df[df["geo"] == region]

    with col2:
        fig = px.line(
            region_df,
            x="year",
            y="unemp_rate",
            title=f"Unemployment Rate in {region_name} (%)"
        )
        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Unemployment Rate (%)"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Geographical Distribution (Latest Year)")

    if GEO_PATH.exists():

        latest_year = df["year"].max()
        latest = df[df["year"] == latest_year]

        with open(GEO_PATH, encoding="utf-8") as f:
            geojson = json.load(f)

        geojson["features"] = [
            feature for feature in geojson["features"]
            if feature["properties"]["NUTS_ID"].startswith("IT")
        ]

        fig_map = px.choropleth(
            latest,
            geojson=geojson,
            locations="geo",
            featureidkey="properties.NUTS_ID",
            color="unemp_rate",
            color_continuous_scale="Blues",
            title=f"Unemployment Rate by Region ({latest_year})"
        )

        fig_map.update_geos(fitbounds="locations", visible=False)
        fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0})

        st.plotly_chart(fig_map, use_container_width=True)

    else:
        st.warning("GeoJSON file not found. Add italy_nuts2.geojson to data/geo/")

# ====================================================
# MODEL EVALUATION
# ====================================================
with tabs[1]:

    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            metrics = json.load(f)

        metrics_df = pd.DataFrame(metrics).T.reset_index()
        metrics_df.rename(columns={"index": "Model"}, inplace=True)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Model Performance Metrics")
            st.dataframe(metrics_df, use_container_width=True)

        with col2:
            best_model = metrics_df.sort_values("RMSE").iloc[0]["Model"]
            st.success("Best Performing Model")
            st.markdown(f"### {best_model}")

# ====================================================
# STRUCTURAL ANALYSIS
# ====================================================
with tabs[2]:

    cluster_df = run_clustering(df)

    st.subheader("Cluster Distribution")
    st.bar_chart(cluster_df["cluster"].value_counts())

    st.subheader("PCA Visualization")

    fig = px.scatter(
        cluster_df,
        x="pca1",
        y="pca2",
        color="cluster",
        hover_name="geo"
    )
    st.plotly_chart(fig, use_container_width=True)

# ====================================================
# FORECAST
# ====================================================
with tabs[3]:

    if PRED_PATH.exists():
        preds = pd.read_csv(PRED_PATH)
        latest_preds = preds[preds["year"] == preds["year"].max()]
        ranked = latest_preds.sort_values("y_pred_next_year", ascending=False)

        st.subheader("Predicted Next-Year Unemployment Ranking")
        st.dataframe(
            ranked[["geo", "model", "y_pred_next_year"]],
            use_container_width=True
        )

# ====================================================
# METHODOLOGY
# ====================================================
with tabs[4]:

    st.title("Project Overview & Methodology")

    st.markdown("""
This project develops a machine learning framework 
for forecasting regional unemployment using panel data.

Objectives:

• Model temporal unemployment dynamics  
• Compare linear and non-linear models  
• Apply time-aware validation  
• Explore structural patterns through clustering  
""")

    st.divider()

    st.header("Problem Formulation")

    st.latex(r"y_{r,t} = \text{UnempRate}_{r,t+1}")
    st.latex(r"\hat{y}_{r,t} = f(x_{r,t})")

    st.markdown("""
Each observation corresponds to a region-year pair.
The model predicts next-year unemployment using current economic indicators.
""")

    st.divider()

    st.header("Feature Engineering")

    st.latex(r"\text{UnempLag1}_{r,t} = \text{UnempRate}_{r,t-1}")
    st.latex(r"\text{GDPLag1}_{r,t} = \text{GDP}_{r,t-1}")
    st.latex(r"\text{GDPGrowth}_{r,t} = 100 \times \frac{\text{GDP}_{r,t} - \text{GDP}_{r,t-1}}{\text{GDP}_{r,t-1}}")

    st.divider()

    st.header("Models")

    # Ridge
    st.subheader("Ridge Regression")

    st.markdown("""
Ridge Regression is a linear model with L2 regularization.
It assumes a linear relationship between economic indicators and unemployment.
Regularization prevents overfitting and stabilizes coefficient estimates.
""")

    st.latex(r"\hat{y} = X\beta")
    st.latex(r"\min_{\beta} ||y - X\beta||_2^2 + \lambda ||\beta||_2^2")

    st.markdown("""
• Interpretable coefficients  
• Stable under correlated predictors  
• Controls variance via λ  
""")

    st.divider()

    # Random Forest
    st.subheader("Random Forest")

    st.markdown("""
Random Forest is a non-linear ensemble model.
It aggregates multiple decision trees trained on random subsets of the data.
""")

    st.latex(r"\hat{y}(x) = \frac{1}{B} \sum_{b=1}^{B} T_b(x)")

    st.markdown("""
• Captures non-linear relationships  
• Handles feature interactions  
• More flexible but less interpretable  

Comparing both models allows evaluation of linear versus non-linear modeling capacity.
""")

    st.divider()

    st.header("Evaluation Metrics")

    st.latex(r"\text{MAE} = \frac{1}{n} \sum |y_i - \hat{y}_i|")
    st.latex(r"\text{RMSE} = \sqrt{\frac{1}{n} \sum (y_i - \hat{y}_i)^2}")
    st.latex(r"R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}")

# ====================================================
# DATA DICTIONARY
# ====================================================
with tabs[5]:

    st.title("Data Dictionary")

    data_dict = pd.DataFrame({
        "Variable": [
            "geo",
            "region",
            "year",
            "unemp_rate",
            "gdp",
            "unemp_rate_lag1",
            "gdp_lag1",
            "gdp_yoy_pct",
            "target_unemp_next_year"
        ],
        "Type": [
            "Categorical",
            "Categorical",
            "Integer",
            "Float",
            "Float",
            "Float",
            "Float",
            "Float",
            "Float"
        ],
        "Description": [
            "Eurostat NUTS2 region code",
            "Region name",
            "Observation year",
            "Annual unemployment rate (%)",
            "Gross Domestic Product (current prices)",
            "Previous year's unemployment rate",
            "Previous year's GDP",
            "GDP year-over-year growth (%)",
            "Next-year unemployment (prediction target)"
        ]
    })

    st.dataframe(data_dict, use_container_width=True)