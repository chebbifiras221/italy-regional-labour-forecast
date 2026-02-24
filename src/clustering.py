from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd



def run_clustering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cluster Italian NUTS2 regions (latest year only)
    using unemployment + GDP.
    Returns dataframe with cluster + PCA coordinates.
    """

    latest_year = df["year"].max()
    latest = df[df["year"] == latest_year].copy()

    features = latest[["unemp_rate", "gdp"]].dropna()
    latest = latest.loc[features.index].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    latest["cluster"] = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)

    latest["pca1"] = components[:, 0]
    latest["pca2"] = components[:, 1]

    return latest