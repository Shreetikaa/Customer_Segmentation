"""
Customer segmentation for the Kaggle "E-Commerce Data" (carrie1/ecommerce-data).

Steps:
1. Data quality cleaning: drop missing CustomerID, cancelled invoices,
   zero/negative quantity or price, duplicate rows
2. Build one row per customer with 3 features:
   - Days_Since_Last_Purchase
   - Number_of_Orders
   - Total_Spend
3. Winsorization: cap each feature at its 1st and 99th percentile so extreme
   customers cannot distort K-Means
4. Normalisation: MinMaxScaler (or StandardScaler) so all 3 features count equally
5. K-Means with K = 3 (checked with the elbow method and silhouette score)
6. Anomaly detection with DBSCAN: customers in no dense region (noise) are anomalies,
   then grouped into 4 anomaly groups by what makes them unusual
"""
import os
import shutil
import zipfile

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ---------------- data folder ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_PATH = os.path.join(DATA_DIR, "online_retail.csv")                # raw Kaggle data
CLEANED_PATH = os.path.join(DATA_DIR, "online_retail_cleaned.csv")    # cleaned data
PROCESSED_PATH = os.path.join(DATA_DIR, "customer_segments.csv")      # processed result

# files from the earlier layout (moved into data/ automatically)
OLD_ZIP = os.path.join(BASE_DIR, "data.csv.zip")
OLD_CSV = os.path.join(BASE_DIR, "data.csv")
OLD_RESULTS = os.path.join(BASE_DIR, "customer_segments.csv")

# ---------------- settings ----------------
K = 3                         # number of clusters (fixed for interpretability)
WINSOR_LIMITS = (0.01, 0.99)  # winsorize at the 1st and 99th percentile
DBSCAN_MIN_SAMPLES = 6        # 2 x number of features (common rule of thumb)

FEATURES = ["Days_Since_Last_Purchase", "Number_of_Orders", "Total_Spend"]
LABELS = {
    "Days_Since_Last_Purchase": "Days since last purchase",
    "Number_of_Orders": "Number of orders",
    "Total_Spend": "Total spend (£)",
}
SEGMENT_NAMES = ["Loyal high-value customers", "Regular customers", "Inactive customers"]
ANOMALY_GROUPS = {
    "both": "Mega buyers",
    "spend": "Big-spend buyers",
    "orders": "Very frequent buyers",
    "other": "Unusual pattern",
}


def setup_data_folders():
    """
    One-time tidy-up: create data/ and put the raw dataset at data/online_retail.csv
    (unzipping data.csv.zip if needed). Old copies are removed once it is safely there.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(RAW_PATH):
        if os.path.exists(OLD_CSV):
            os.replace(OLD_CSV, RAW_PATH)
        elif os.path.exists(OLD_ZIP):
            with zipfile.ZipFile(OLD_ZIP) as z:
                name = [n for n in z.namelist() if n.endswith(".csv")][0]
                with z.open(name) as src, open(RAW_PATH, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    if os.path.exists(RAW_PATH) and os.path.getsize(RAW_PATH) > 0:
        for old in (OLD_ZIP, OLD_CSV, OLD_RESULTS,
                    os.path.join(DATA_DIR, "processed", "customer_segments.csv")):
            if os.path.exists(old):
                os.remove(old)
        for sub in ("raw", "cleaned", "processed"):      # empty leftover sub-folders
            p = os.path.join(DATA_DIR, sub)
            if os.path.isdir(p) and not os.listdir(p):
                os.rmdir(p)


def find_raw_data():
    """Return data/online_retail.csv (setting up the data folder first if needed)."""
    setup_data_folders()
    return RAW_PATH if os.path.exists(RAW_PATH) else None


def save_outputs(out):
    """Save cleaned transactions and customer segments into data/."""
    os.makedirs(DATA_DIR, exist_ok=True)
    out["transactions"].to_csv(CLEANED_PATH, index=False)
    out["customers"].to_csv(PROCESSED_PATH)


# ---------------- step 1: load and clean ----------------
def load_data(path):
    df = pd.read_csv(path, encoding="latin1",
                     dtype={"CustomerID": str, "InvoiceNo": str, "StockCode": str})
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


def clean_data(df):
    d = df.dropna(subset=["CustomerID"])
    d = d[~d["InvoiceNo"].str.startswith("C")]          # cancelled invoices
    d = d[(d["Quantity"] > 0) & (d["UnitPrice"] > 0)]
    d = d.drop_duplicates().copy()
    d["CustomerID"] = d["CustomerID"].str.replace(r"\.0$", "", regex=True)
    d["TotalPrice"] = d["Quantity"] * d["UnitPrice"]
    return d


# ---------------- step 2: customer features ----------------
def build_customer_table(d):
    last_date = d["InvoiceDate"].max() + pd.Timedelta(days=1)
    cust = d.groupby("CustomerID").agg(
        Days_Since_Last_Purchase=("InvoiceDate", lambda x: (last_date - x.max()).days),
        Number_of_Orders=("InvoiceNo", "nunique"),
        Total_Spend=("TotalPrice", "sum"),
    )
    cust["Country"] = d.groupby("CustomerID")["Country"].agg(lambda x: x.mode().iat[0])
    return cust


# ---------------- step 3: winsorization ----------------
def winsorize(features, limits=WINSOR_LIMITS):
    """Cap every value below the 1st percentile / above the 99th percentile at that percentile."""
    lower = features.quantile(limits[0])
    upper = features.quantile(limits[1])
    capped = features.clip(lower=lower, upper=upper, axis=1)
    caps = pd.DataFrame({"Lower cap (1st percentile)": lower,
                         "Upper cap (99th percentile)": upper,
                         "Customers capped": ((features < lower) | (features > upper)).sum()})
    return capped, caps


# ---------------- step 4: normalisation ----------------
def scale(features, method="MinMax"):
    """MinMaxScaler -> every feature between 0 and 1; StandardScaler -> mean 0, standard deviation 1."""
    scaler = MinMaxScaler() if method == "MinMax" else StandardScaler()
    return pd.DataFrame(scaler.fit_transform(features), index=features.index, columns=features.columns)


# ---------------- step 5: checking K + K-Means ----------------
def evaluate_k(X, k_range=range(2, 11)):
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        rows.append({"K": k, "Inertia": km.inertia_,
                     "Silhouette": silhouette_score(X, km.labels_)})
    return pd.DataFrame(rows)


def elbow_k(scores):
    """
    Elbow = the K where the drop in inertia slows down the most
    (largest change between one drop and the next, i.e. the sharpest bend).
    """
    drops = scores["Inertia"].diff()          # how much inertia falls from K-1 to K (negative numbers)
    bend = drops.diff()                       # how much smaller the next drop is
    return int(scores["K"].iloc[bend.idxmax() - 1])


def name_clusters(profile):
    """
    Name the 3 clusters from their average features:
    highest spend -> Loyal high-value, most days since last purchase -> Inactive, the other -> Regular.
    """
    high = profile["Total_Spend"].idxmax()
    inactive = profile.drop(index=high)["Days_Since_Last_Purchase"].idxmax()
    names = {}
    for c in profile.index:
        if c == high:
            names[c] = SEGMENT_NAMES[0]
        elif c == inactive:
            names[c] = SEGMENT_NAMES[2]
        else:
            names[c] = SEGMENT_NAMES[1]
    return names


# ---------------- step 6: DBSCAN anomalies ----------------
def k_distance(X, min_samples=DBSCAN_MIN_SAMPLES):
    """Distance from each customer to its min_samples-th nearest neighbour, sorted (to choose eps)."""
    nn = NearestNeighbors(n_neighbors=min_samples).fit(X)
    return np.sort(nn.kneighbors(X)[0][:, -1])


def knee_eps(dist):
    """eps = the 'knee' of the k-distance curve: the point farthest below the line from first to last point."""
    x = np.linspace(0, 1, len(dist))
    y = (dist - dist.min()) / (dist.max() - dist.min())
    return float(dist[np.argmax(x - y)])


def dbscan_anomalies(features, min_samples=DBSCAN_MIN_SAMPLES):
    """
    DBSCAN on the standardised (not winsorized) features.
    Customers that are not in any dense region get label -1 = noise = anomaly.
    """
    X = StandardScaler().fit_transform(features)
    dist = k_distance(X, min_samples)
    eps = knee_eps(dist)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    return labels == -1, eps, dist


def anomaly_groups(features, is_anomaly):
    """Group the DBSCAN anomalies by what makes them unusual (compared with the 99th percentile)."""
    p99 = features.quantile(WINSOR_LIMITS[1])
    a = features[is_anomaly]
    high_spend = a["Total_Spend"] > p99["Total_Spend"]
    high_orders = a["Number_of_Orders"] > p99["Number_of_Orders"]
    group = np.select([high_spend & high_orders, high_spend, high_orders],
                      [ANOMALY_GROUPS["both"], ANOMALY_GROUPS["spend"], ANOMALY_GROUPS["orders"]],
                      default=ANOMALY_GROUPS["other"])
    return pd.Series(group, index=a.index, dtype=object)


# ---------------- the whole pipeline ----------------
def run_pipeline(df, scaler="MinMax", k=K):
    d = clean_data(df)                                   # step 1
    cust = build_customer_table(d)                       # step 2
    feats = cust[FEATURES]

    capped, caps = winsorize(feats)                      # step 3
    X = scale(capped, scaler)                            # step 4

    scores = evaluate_k(X)                               # step 5
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    sil = silhouette_score(X, km.labels_)
    profile = feats.groupby(km.labels_).mean()
    cust["Segment"] = pd.Series(km.labels_, index=cust.index).map(name_clusters(profile))

    is_anom, eps, kdist = dbscan_anomalies(feats)        # step 6
    cust["IsAnomaly"] = is_anom
    cust["AnomalyGroup"] = anomaly_groups(feats, is_anom)

    return {"transactions": d, "customers": cust, "capped": capped, "scaled": X,
            "winsor_caps": caps, "k_scores": scores, "k": k, "silhouette": sil,
            "eps": eps, "k_distance": kdist, "scaler": scaler}


if __name__ == "__main__":
    raw = find_raw_data()
    if raw is None:
        raise SystemExit("Raw data not found: put the Kaggle file in data/ as online_retail.csv")
    print("Reading raw data from", raw)
    out = run_pipeline(load_data(raw))
    c = out["customers"]

    print("\nWinsorization caps:\n", out["winsor_caps"].round(1))
    print("\nK scores:\n", out["k_scores"].round(3).to_string(index=False))
    print(f"\nK = {out['k']} | silhouette = {out['silhouette']:.3f} | scaler = {out['scaler']}")
    print(c.groupby("Segment")[FEATURES].agg(["mean", "count"]).round(1).to_string())
    print(f"\nDBSCAN eps = {out['eps']:.3f} | anomalies = {int(c['IsAnomaly'].sum())}")
    print(c["AnomalyGroup"].value_counts().to_string())

    save_outputs(out)
    print(f"\nsaved cleaned data   -> {CLEANED_PATH} ({len(out['transactions']):,} rows)")
    print(f"saved processed data -> {PROCESSED_PATH} ({len(c):,} customers)")
