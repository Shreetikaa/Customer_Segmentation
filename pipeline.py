"""
Customer segmentation for the Kaggle "E-Commerce Data" (carrie1/ecommerce-data).

Steps (all methods from the course):
1. Data quality cleaning (Week 1): drop missing CustomerID, cancelled invoices,
   zero/negative quantity or price, duplicate rows
2. Build one row per customer with 3 features:
   - Days_Since_Last_Purchase
   - Number_of_Orders
   - Total_Spend
3. Outliers / anomalies (Week 8): IQR rule on Number_of_Orders and Total_Spend,
   then anomalies split into 4 groups by why they are outliers (1.5×IQR and 3×IQR limits)
4. Choose K with the Elbow method and Silhouette score (Week 4 - evaluation metrics)
5. K-Means clustering on the normal customers (Week 4)
"""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

FEATURES = ["Days_Since_Last_Purchase", "Number_of_Orders", "Total_Spend"]
OUTLIER_FEATURES = ["Number_of_Orders", "Total_Spend"]
LABELS = {
    "Days_Since_Last_Purchase": "Days since last purchase",
    "Number_of_Orders": "Number of orders",
    "Total_Spend": "Total spend (£)",
}
ANOMALY_LABEL = "Anomalies (outliers)"


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


def build_customer_table(d):
    last_date = d["InvoiceDate"].max() + pd.Timedelta(days=1)
    cust = d.groupby("CustomerID").agg(
        Days_Since_Last_Purchase=("InvoiceDate", lambda x: (last_date - x.max()).days),
        Number_of_Orders=("InvoiceNo", "nunique"),
        Total_Spend=("TotalPrice", "sum"),
    )
    cust["Country"] = d.groupby("CustomerID")["Country"].agg(lambda x: x.mode().iat[0])
    return cust


ANOMALY_GROUPS = {
    "extreme": "Anomaly – Extreme bulk buyers",
    "both": "Anomaly – Frequent big spenders",
    "spend": "Anomaly – Big-order buyers",
    "orders": "Anomaly – Very frequent small buyers",
}


def iqr_outliers(cust):
    """
    IQR rule: a value is an outlier if it is above Q3 + 1.5*IQR (or below Q1 - 1.5*IQR).
    Values above Q3 + 3*IQR are 'extreme' outliers (Tukey's rule).

    Anomalies are then put into 4 groups by WHY they are outliers:
      1. Extreme bulk buyers        - extreme (> 3*IQR) on BOTH orders and spend
      2. Frequent big spenders      - outlier on both orders and spend
      3. Big-order buyers           - outlier on spend only (few but very large orders)
      4. Very frequent small buyers - outlier on orders only (many orders, normal spend)
    """
    q1 = cust[OUTLIER_FEATURES].quantile(0.25)
    q3 = cust[OUTLIER_FEATURES].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    extreme = q3 + 3 * iqr
    high = cust[OUTLIER_FEATURES] > upper
    low = cust[OUTLIER_FEATURES] < lower
    ext = cust[OUTLIER_FEATURES] > extreme
    is_outlier = (high | low).any(axis=1)

    def group(cid):
        if ext.loc[cid].all():
            return ANOMALY_GROUPS["extreme"]
        if high.loc[cid].all():
            return ANOMALY_GROUPS["both"]
        if high.loc[cid, "Total_Spend"]:
            return ANOMALY_GROUPS["spend"]
        return ANOMALY_GROUPS["orders"]

    def reason(cid):
        r = []
        if high.loc[cid, "Total_Spend"]:
            r.append("extreme spend" if ext.loc[cid, "Total_Spend"] else "very high spend")
        if high.loc[cid, "Number_of_Orders"]:
            r.append("extremely frequent" if ext.loc[cid, "Number_of_Orders"]
                     else "very frequent buyer")
        return " + ".join(r)

    ids = cust.index[is_outlier]
    groups = pd.Series([group(c) for c in ids], index=ids, dtype=object)
    reasons = pd.Series([reason(c) for c in ids], index=ids, dtype=object)
    bounds = pd.DataFrame({"Q1": q1, "Q3": q3, "IQR": iqr, "Lower limit": lower,
                           "Upper limit (Q3 + 1.5×IQR)": upper,
                           "Extreme limit (Q3 + 3×IQR)": extreme})
    return is_outlier, reasons, groups, bounds


def evaluate_k(X, k_range=range(2, 11)):
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        rows.append({"K": k, "Inertia": km.inertia_,
                     "Silhouette": silhouette_score(X, km.labels_)})
    return pd.DataFrame(rows)


def elbow_k(scores):
    """Elbow = the K furthest from the straight line joining the first and last inertia points."""
    k = scores["K"].astype(float)
    y = scores["Inertia"].astype(float)
    kn = (k - k.min()) / (k.max() - k.min())
    yn = (y - y.min()) / (y.max() - y.min())
    dist = (kn + yn - 1).abs()
    return int(scores["K"].iloc[dist.values.argmax()])


def recommend_k(scores):
    """
    K=2 usually has the highest silhouette but only splits customers into two groups,
    which is too coarse. So: take the elbow point and, around it (elbow-1 .. elbow+1),
    choose the K with the best silhouette.
    """
    ek = elbow_k(scores)
    s = scores.set_index("K")["Silhouette"]
    region = s[(s.index >= max(3, ek - 1)) & (s.index <= ek + 1)]
    return {"elbow_k": ek, "best_silhouette_k": int(s.idxmax()),
            "recommended_k": int(region.idxmax())}


TIERS = {
    2: ["High spenders", "Low spenders"],
    3: ["High spenders", "Medium spenders", "Low spenders"],
    4: ["High spenders", "Medium-high spenders", "Medium-low spenders", "Low spenders"],
    5: ["Top spenders", "High spenders", "Medium spenders", "Low spenders", "Lowest spenders"],
}


def name_clusters(profile):
    """Name clusters by average total spend (highest first)."""
    order = profile["Total_Spend"].sort_values(ascending=False).index
    names = TIERS.get(len(order), [f"Group {i + 1}" for i in range(len(order))])
    return {c: n for c, n in zip(order, names)}


def run_pipeline(df, k=None):
    d = clean_data(df)
    cust = build_customer_table(d)

    is_out, reasons, groups, bounds = iqr_outliers(cust)
    cust["IsAnomaly"] = is_out
    cust["AnomalyReason"] = reasons
    cust["AnomalyGroup"] = groups

    X = cust.loc[~is_out, FEATURES]
    scores = evaluate_k(X)
    rec = recommend_k(scores)
    k = k or rec["recommended_k"]

    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    cust["Cluster"] = -1
    cust.loc[~is_out, "Cluster"] = km.labels_
    sil = silhouette_score(X, km.labels_)

    profile = X.groupby(km.labels_).mean()
    names = name_clusters(profile)
    cust["Segment"] = cust["Cluster"].map(names).fillna(cust["AnomalyGroup"])

    return {"transactions": d, "customers": cust, "k_scores": scores,
            "recommendation": rec, "k": k, "silhouette": sil, "iqr_bounds": bounds}


if __name__ == "__main__":
    import os
    path = "data.csv" if os.path.exists("data.csv") else "data.csv.zip"
    out = run_pipeline(load_data(path))
    print(out["iqr_bounds"].round(1))
    print(out["k_scores"].round(3).to_string(index=False))
    print(out["recommendation"], "-> K =", out["k"], "| silhouette =", round(out["silhouette"], 3))
    c = out["customers"]
    print(c.groupby("Segment")[FEATURES].agg(["mean", "count"]).round(1).to_string())
    print(c["AnomalyGroup"].value_counts())
    c.to_csv("customer_segments.csv")
    print("saved customer_segments.csv")
