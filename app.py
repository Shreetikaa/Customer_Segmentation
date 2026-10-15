"""
Streamlit app: Customer Segmentation (winsorization + scaling + K-Means, K = 3)
and Anomaly Detection (DBSCAN)
Run:  streamlit run app.py
"""
import os

import pandas as pd
import plotly.express as px
import streamlit as st

from pipeline import (ANOMALY_GROUPS, CLEANED_PATH, DBSCAN_MIN_SAMPLES, FEATURES, LABELS,
                      PROCESSED_PATH, SEGMENT_NAMES, elbow_k, find_raw_data, load_data,
                      run_pipeline, save_outputs)

st.set_page_config(page_title="Customer Segmentation", page_icon="🛒", layout="wide")

COLORS = {
    "Loyal high-value customers": "#2E7D32",
    "Regular customers": "#1565C0",
    "Inactive customers": "#9E9E9E",
    ANOMALY_GROUPS["both"]: "#B71C1C",
    ANOMALY_GROUPS["spend"]: "#FB8C00",
    ANOMALY_GROUPS["orders"]: "#8E24AA",
    ANOMALY_GROUPS["other"]: "#6D4C41",
    "Normal customers": "#BDBDBD",
}

ACTIONS = {
    "Loyal high-value customers": "Reward and retain: loyalty programme, early access, personal offers.",
    "Regular customers": "Grow them: product bundles and recommendations to raise spend and frequency.",
    "Inactive customers": "Win back: 'we miss you' discount and reminder emails.",
    ANOMALY_GROUPS["both"]: "Key accounts (likely wholesalers): account manager and volume pricing.",
    ANOMALY_GROUPS["spend"]: "Very large orders: check one-off bulk orders or data errors; offer repeat deals.",
    ANOMALY_GROUPS["orders"]: "Order very often: subscription or minimum-order discounts.",
    ANOMALY_GROUPS["other"]: "Unusual mix of values: review manually.",
}
SEG_ORDER = SEGMENT_NAMES
ANOM_ORDER = list(ANOMALY_GROUPS.values())


def action_for(group):
    return ACTIONS.get(group, "")


@st.cache_data(show_spinner="Loading data…")
def get_data(path):
    return load_data(path)


@st.cache_data(show_spinner="Running winsorization, K-Means and DBSCAN…")
def get_results(_df, data_key):
    out = run_pipeline(_df)
    # keep data/ up to date: processed results every run, cleaned data only if missing
    out["customers"].to_csv(PROCESSED_PATH)
    if not os.path.exists(CLEANED_PATH):
        save_outputs(out)
    return out


data_path = find_raw_data()          # data/online_retail.csv
if data_path is None:
    st.title("🛒 Customer Segmentation & Anomaly Detection")
    st.info("Put the Kaggle file (carrie1/ecommerce-data) in the **data/** folder as **online_retail.csv**.")
    st.stop()
df = get_data(data_path)

# ---------------- sidebar ----------------
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("**Normalisation:** MinMaxScaler")
st.sidebar.markdown("**Number of clusters:** K = 3 (fixed)")
res = get_results(df, data_path)

cust = res["customers"]
anom = cust[cust["IsAnomaly"]]
anom_groups = [g for g in ANOM_ORDER if g in set(anom["AnomalyGroup"])]

st.sidebar.markdown("---")
st.sidebar.caption("Steps: clean data → customer features → winsorization (1st–99th percentile) → "
                   "MinMaxScaler → K-Means (K = 3) · DBSCAN for anomalies")

# ---------------- header ----------------
st.title("🛒 Customer Segmentation & Anomaly Detection")
st.caption("E-commerce transactions (UK online retailer, Dec 2010 – Dec 2011) · "
           "Winsorization · Normalisation · K-Means (K = 3) · DBSCAN anomaly detection")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", f"{len(cust):,}")
c2.metric("Clusters (K)", res["k"])
c3.metric("Silhouette score", f"{res['silhouette']:.3f}")
c4.metric("DBSCAN anomalies", f"{len(anom):,}")

tabs = st.tabs(["Features & Preprocessing", "Choosing K", "Clusters",
                "Anomalies (DBSCAN)", "Customer lookup"])

# ---------------- tab 1: features + winsorization + scaling ----------------
with tabs[0]:
    st.subheader("1. Customer features (one row per customer)")
    st.markdown("- **Days since last purchase** – how recently the customer bought\n"
                "- **Number of orders** – number of different invoices\n"
                "- **Total spend (£)** – sum of Quantity × UnitPrice")
    st.dataframe(cust[FEATURES].rename(columns=LABELS).describe().round(1), width="stretch")

    st.subheader("2. Winsorization – managing outliers")
    st.markdown("Each feature is **capped at its 1st and 99th percentile**: values above the 99th "
                "percentile are replaced by the 99th-percentile value (and the same at the bottom). "
                "Customers are kept, but a few extreme values can no longer pull the K-Means centres.")
    st.dataframe(res["winsor_caps"].rename(index=LABELS).style.format(
        {"Lower cap (1st percentile)": "{:,.1f}", "Upper cap (99th percentile)": "{:,.1f}",
         "Customers capped": "{:,.0f}"}), width="stretch")
    feat = st.radio("Show feature", FEATURES, index=2, format_func=LABELS.get,
                    horizontal=True, key="wfeat")
    before, after = cust[feat], res["capped"][feat]
    m1, m2, m3 = st.columns(3)
    m1.metric("Max before", f"{before.max():,.0f}")
    m2.metric("Max after", f"{after.max():,.0f}",
              delta=f"{after.max() - before.max():,.0f}", delta_color="off")
    m3.metric("Customers capped", f"{int((before != after).sum()):,}")
    a, b = st.columns(2)          # separate charts, each with its own y-axis
    a.plotly_chart(px.box(y=before, points="outliers", color_discrete_sequence=["#9E9E9E"],
                          labels={"y": LABELS[feat]}, title="Before winsorization"),
                   width="stretch")
    b.plotly_chart(px.box(y=after, points="outliers", color_discrete_sequence=["#1565C0"],
                          labels={"y": LABELS[feat]}, title="After winsorization"),
                   width="stretch")
    st.caption("Each chart has its own scale. Before: a few extreme customers stretch the axis and "
               "squash everyone else. After: values above the 99th percentile are capped, so the "
               "box and whiskers are visible. Days since last purchase changes very little "
               "(374 → 369) because its values are not very extreme.")

    st.subheader("3. Normalisation – MinMaxScaler")
    st.markdown("After winsorization every feature is scaled with **MinMaxScaler**: "
                "**(value − min) ÷ (max − min)**, so every feature goes from **0 to 1**. "
                "Days (1–369), orders (1–30) and spend (£52–£19,780) then count **equally** in "
                "K-Means. Without it, spend would dominate only because its numbers are bigger.")
    ranges = pd.DataFrame({
        "Min before": res["capped"].min(), "Max before": res["capped"].max(),
        "Min after": res["scaled"].min(), "Max after": res["scaled"].max()}).rename(index=LABELS)
    st.dataframe(ranges.style.format({"Min before": "{:,.1f}", "Max before": "{:,.1f}",
                                      "Min after": "{:.1f}", "Max after": "{:.1f}"}),
                 width="stretch")
    short = {"Days_Since_Last_Purchase": "Days since last purchase",
             "Number_of_Orders": "Number of orders", "Total_Spend": "Total spend"}
    long_before = res["capped"].rename(columns=short).melt(var_name="Feature", value_name="Value")
    long_after = res["scaled"].rename(columns=short).melt(var_name="Feature", value_name="Value")
    a, b = st.columns(2)
    a.plotly_chart(px.box(long_before, x="Feature", y="Value", color="Feature",
                          title="Before normalisation (winsorized values)")
                   .update_layout(showlegend=False, xaxis_title=None), width="stretch")
    b.plotly_chart(px.box(long_after, x="Feature", y="Value", color="Feature",
                          title="After normalisation (MinMaxScaler, 0 – 1)")
                   .update_layout(showlegend=False, xaxis_title=None), width="stretch")
    st.caption("Before: spend (up to about £20,000) is so much bigger that days and orders look "
               "like flat lines at 0, so K-Means distances would be decided by spend alone. "
               "After: all 3 features are on the same 0 – 1 scale and can be compared fairly.")

# ---------------- tab 2: choosing K ----------------
with tabs[1]:
    sc = res["k_scores"]
    ek = elbow_k(sc)
    best_sil = int(sc.loc[sc["Silhouette"].idxmax(), "K"])
    sil3 = sc.set_index("K").loc[3, "Silhouette"]
    st.subheader("Why K = 3")
    a, b = st.columns(2)
    fig = px.line(sc, x="K", y="Inertia", markers=True, title="Elbow method (inertia / WCSS)")
    fig.add_vline(x=3, line_dash="dash", line_color="#2E7D32", annotation_text="K = 3")
    a.plotly_chart(fig, width="stretch")
    fig = px.line(sc, x="K", y="Silhouette", markers=True, title="Silhouette score (higher is better)")
    fig.add_vline(x=3, line_dash="dash", line_color="#2E7D32", annotation_text="K = 3")
    b.plotly_chart(fig, width="stretch")
    st.dataframe(sc.style.format({"Inertia": "{:,.1f}", "Silhouette": "{:.3f}"})
                 .highlight_max(subset=["Silhouette"], color="#C8E6C9"),
                 width="stretch", hide_index=True)
    st.info(
        f"**K = 3** is used for interpretability: three groups a shop can act on "
        f"(loyal high-value, regular, inactive). The data supports it: the elbow is at **K = {ek}**, "
        f"and the silhouette at K = 3 is **{sil3:.3f}** "
        + ("(the highest of all K)." if best_sil == 3 else f"(the highest is at K = {best_sil}, "
           "but that gives fewer, less useful groups).")
    )
    st.caption("Computed on winsorized, MinMax-scaled features.")

# ---------------- tab 3: clusters ----------------
with tabs[2]:
    st.subheader("Customer segments (K-Means, K = 3)")
    summary = (cust.groupby("Segment")
               .agg(Customers=("Total_Spend", "size"),
                    Avg_days_since_last_purchase=("Days_Since_Last_Purchase", "mean"),
                    Avg_orders=("Number_of_Orders", "mean"),
                    Avg_spend=("Total_Spend", "mean"),
                    Total_revenue=("Total_Spend", "sum"))
               .reindex(SEG_ORDER))
    summary["Share_of_customers_%"] = 100 * summary["Customers"] / summary["Customers"].sum()
    summary["Share_of_revenue_%"] = 100 * summary["Total_revenue"] / summary["Total_revenue"].sum()
    summary["Suggested action"] = [action_for(s) for s in summary.index]
    st.dataframe(summary.style.format({
        "Avg_days_since_last_purchase": "{:.0f}", "Avg_orders": "{:.1f}", "Avg_spend": "£{:,.0f}",
        "Total_revenue": "£{:,.0f}", "Share_of_customers_%": "{:.1f}%",
        "Share_of_revenue_%": "{:.1f}%"}), width="stretch")

    a, b = st.columns(2)
    a.plotly_chart(px.pie(summary.reset_index(), names="Segment", values="Customers",
                          color="Segment", color_discrete_map=COLORS, hole=0.45,
                          title="Share of customers"), width="stretch")
    b.plotly_chart(px.pie(summary.reset_index(), names="Segment", values="Total_revenue",
                          color="Segment", color_discrete_map=COLORS, hole=0.45,
                          title="Share of revenue"), width="stretch")

    view = res["capped"].join(cust[["Segment", "Country"]]).reset_index()
    st.markdown("#### 3D view of the clusters (winsorized values)")
    fig = px.scatter_3d(view, x="Days_Since_Last_Purchase", y="Number_of_Orders",
                        z="Total_Spend", color="Segment", color_discrete_map=COLORS,
                        category_orders={"Segment": SEG_ORDER}, labels=LABELS,
                        opacity=0.7, hover_data=["CustomerID", "Country"], height=650)
    fig.update_traces(marker=dict(size=3))
    st.plotly_chart(fig, width="stretch")

    a, b = st.columns(2)
    a.plotly_chart(px.scatter(view, x="Days_Since_Last_Purchase", y="Total_Spend", color="Segment",
                              color_discrete_map=COLORS, category_orders={"Segment": SEG_ORDER},
                              labels=LABELS, opacity=0.6, hover_data=["CustomerID"],
                              title="Days since last purchase vs total spend"), width="stretch")
    b.plotly_chart(px.scatter(view, x="Number_of_Orders", y="Total_Spend", color="Segment",
                              color_discrete_map=COLORS, category_orders={"Segment": SEG_ORDER},
                              labels=LABELS, opacity=0.6, hover_data=["CustomerID"],
                              title="Number of orders vs total spend"), width="stretch")

    st.markdown("#### Distribution of each feature by segment")
    feat = st.radio("Feature", FEATURES, format_func=LABELS.get, horizontal=True, key="cfeat")
    st.plotly_chart(px.box(view, x="Segment", y=feat, color="Segment", labels=LABELS,
                           color_discrete_map=COLORS, category_orders={"Segment": SEG_ORDER})
                    .update_layout(showlegend=False), width="stretch")

# ---------------- tab 4: DBSCAN anomalies ----------------
with tabs[3]:
    st.subheader(f"{len(anom)} anomalous customers found by DBSCAN")
    st.markdown(
        "**DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) groups customers "
        "that sit in **dense areas**. A customer with too few neighbours nearby belongs to no dense "
        "area and is labelled **noise**; these are our anomalies. It runs on the MinMax-scaled "
        "features **before** winsorization, so the extreme values are still visible to it.")
    a, b, c, d = st.columns(4)
    a.metric("eps (neighbourhood radius)", f"{res['eps']:.3f}")
    b.metric("min_samples", DBSCAN_MIN_SAMPLES)
    c.metric("Share of customers", f"{100 * len(anom) / len(cust):.1f}%")
    d.metric("Share of revenue", f"{100 * anom['Total_Spend'].sum() / cust['Total_Spend'].sum():.1f}%")

    kd = pd.DataFrame({"Customers (sorted)": range(1, len(res["k_distance"]) + 1),
                       "Distance to 6th nearest neighbour": res["k_distance"]})
    fig = px.line(kd, x="Customers (sorted)", y="Distance to 6th nearest neighbour",
                  title="k-distance plot: eps is chosen at the knee of the curve")
    fig.add_hline(y=res["eps"], line_dash="dash", line_color="#B71C1C",
                  annotation_text=f"eps = {res['eps']:.3f}")
    st.plotly_chart(fig, width="stretch")
    st.caption("min_samples = 6 (2 × the 3 features). Customers whose 6th neighbour is farther "
               "than eps are not in a dense area → noise → anomaly.")

    st.markdown("#### 4 groups of anomalies")
    st.markdown("Each DBSCAN anomaly is grouped by **what makes it unusual**, compared with the "
                "99th percentile of all customers:\n"
                f"1. **{ANOMALY_GROUPS['both']}** – above the 99th percentile on **both** spend and orders\n"
                f"2. **{ANOMALY_GROUPS['spend']}** – above it on **spend only**\n"
                f"3. **{ANOMALY_GROUPS['orders']}** – above it on **orders only**\n"
                f"4. **{ANOMALY_GROUPS['other']}** – an unusual **combination** of values")
    gsum = (anom.groupby("AnomalyGroup")
            .agg(Customers=("Total_Spend", "size"),
                 Avg_days_since_last_purchase=("Days_Since_Last_Purchase", "mean"),
                 Avg_orders=("Number_of_Orders", "mean"),
                 Avg_spend=("Total_Spend", "mean"),
                 Total_revenue=("Total_Spend", "sum"))
            .reindex(anom_groups))
    gsum["Share_of_total_revenue_%"] = 100 * gsum["Total_revenue"] / cust["Total_Spend"].sum()
    gsum["Suggested action"] = [action_for(g) for g in gsum.index]
    st.dataframe(gsum.style.format({
        "Avg_days_since_last_purchase": "{:.0f}", "Avg_orders": "{:.1f}", "Avg_spend": "£{:,.0f}",
        "Total_revenue": "£{:,.0f}", "Share_of_total_revenue_%": "{:.1f}%"}), width="stretch")

    plot_df = cust.reset_index()
    plot_df["Group"] = plot_df["AnomalyGroup"].where(plot_df["IsAnomaly"], "Normal customers")
    fig = px.scatter(plot_df, x="Number_of_Orders", y="Total_Spend", color="Group",
                     color_discrete_map=COLORS,
                     category_orders={"Group": ["Normal customers"] + anom_groups},
                     log_x=True, log_y=True, opacity=0.75, labels=LABELS,
                     hover_data=["CustomerID", "Days_Since_Last_Purchase", "Segment"], height=550,
                     title="DBSCAN anomalies vs normal customers (log axes for display only)")
    st.plotly_chart(fig, width="stretch")

    a, b = st.columns(2)
    a.plotly_chart(px.bar(gsum.reset_index(), x="Customers", y="AnomalyGroup", orientation="h",
                          color="AnomalyGroup", color_discrete_map=COLORS, text="Customers",
                          category_orders={"AnomalyGroup": anom_groups},
                          title="Customers in each anomaly group")
                   .update_layout(showlegend=False, yaxis_title=None), width="stretch")
    b.plotly_chart(px.bar(gsum.reset_index(), x="Total_revenue", y="AnomalyGroup", orientation="h",
                          color="AnomalyGroup", color_discrete_map=COLORS,
                          category_orders={"AnomalyGroup": anom_groups},
                          labels={"Total_revenue": "Total revenue (£)"},
                          title="Revenue from each anomaly group")
                   .update_layout(showlegend=False, yaxis_title=None), width="stretch")

    pick = st.selectbox("Show customers in group", ["All anomaly groups"] + anom_groups)
    shown = anom if pick == "All anomaly groups" else anom[anom["AnomalyGroup"] == pick]
    st.dataframe(shown.sort_values("Total_Spend", ascending=False)
                 [["AnomalyGroup", "Segment", "Country"] + FEATURES]
                 .rename(columns={**LABELS, "AnomalyGroup": "Anomaly group", "Segment": "K-Means segment"})
                 .style.format({LABELS["Total_Spend"]: "£{:,.0f}"}), width="stretch")

# ---------------- tab 5: lookup ----------------
with tabs[4]:
    st.subheader("Look up a customer")
    cid = st.selectbox("CustomerID", cust.index.sort_values())
    row = cust.loc[cid]
    st.markdown(f"**Segment:** {row['Segment']}")
    b, c, d = st.columns(3)
    b.metric("Days since last purchase", f"{row['Days_Since_Last_Purchase']}")
    c.metric("Number of orders", f"{row['Number_of_Orders']}")
    d.metric("Total spend", f"£{row['Total_Spend']:,.0f}")
    st.success(f"**Suggested action:** {action_for(row['Segment'])}")
    if row["IsAnomaly"]:
        st.warning(f"**DBSCAN anomaly – {row['AnomalyGroup']}.** {action_for(row['AnomalyGroup'])}")

    tx = res["transactions"]
    st.markdown("Recent transactions")
    st.dataframe(tx[tx["CustomerID"] == cid].sort_values("InvoiceDate", ascending=False).head(50),
                 width="stretch", hide_index=True)

    st.markdown("---")
    seg_filter = st.multiselect("Filter export by segment", SEG_ORDER, default=SEG_ORDER)
    out = cust[cust["Segment"].isin(seg_filter)]
    st.download_button("⬇️ Download customer segments (CSV)", out.to_csv().encode(),
                       "customer_segments.csv", "text/csv")
