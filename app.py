"""
Streamlit app: Customer Segmentation (K-Means) + Anomalies (IQR outliers)
Run:  streamlit run app.py
"""
import os

import pandas as pd
import plotly.express as px
import streamlit as st

from pipeline import (ANOMALY_GROUPS, CLEANED_PATH, FEATURES, LABELS, PROCESSED_PATH,
                      find_raw_data, load_data, run_pipeline, save_outputs)

st.set_page_config(page_title="Customer Segmentation", page_icon="🛒", layout="wide")

ANOMALY_ORDER = list(ANOMALY_GROUPS.values())

SEGMENT_COLORS = {
    "Top spenders": "#1B5E20",
    "High spenders": "#2E7D32",
    "Medium-high spenders": "#1565C0",
    "Medium spenders": "#1565C0",
    "Medium-low spenders": "#42A5F5",
    "Low spenders": "#9E9E9E",
    "Lowest spenders": "#BDBDBD",
    ANOMALY_GROUPS["extreme"]: "#B71C1C",
    ANOMALY_GROUPS["both"]: "#E53935",
    ANOMALY_GROUPS["spend"]: "#FB8C00",
    ANOMALY_GROUPS["orders"]: "#8E24AA",
}

ACTIONS = {
    "Top spenders": "Reward them: loyalty program, early access to new products.",
    "High spenders": "Keep them happy: loyalty rewards and personalised offers.",
    "Medium-high spenders": "Upsell: product bundles and recommendations to increase spend.",
    "Medium spenders": "Upsell: product bundles and recommendations to increase spend.",
    "Medium-low spenders": "Upsell: product bundles and recommendations to increase spend.",
    "Low spenders": "Re-engage: discount on next purchase, reminder emails.",
    "Lowest spenders": "Low-cost reactivation emails only.",
    ANOMALY_GROUPS["extreme"]: "Most valuable accounts (likely wholesalers): dedicated account manager, "
                               "bulk/volume pricing. Also verify the data is genuine.",
    ANOMALY_GROUPS["both"]: "Loyal business buyers: wholesale discounts, priority delivery.",
    ANOMALY_GROUPS["spend"]: "Few but very large orders: check for one-off bulk/event orders "
                             "or possible errors; offer repeat-order deals.",
    ANOMALY_GROUPS["orders"]: "Order often but in small amounts: bundle offers / minimum-order "
                              "discounts to raise order size.",
}


def action_for(segment):
    return ACTIONS.get(segment, "")


@st.cache_data(show_spinner="Loading data…")
def get_data(path):
    return load_data(path)


@st.cache_data(show_spinner="Running K-Means…")
def get_results(_df, k, data_key):
    return run_pipeline(_df, k=k)


data_path = find_raw_data()          # data/online_retail.csv
if data_path is None:
    st.title("🛒 Customer Segmentation & Anomaly Detection")
    st.info("Put the Kaggle file (carrie1/ecommerce-data) in the **data/** folder as **online_retail.csv**.")
    st.stop()
df = get_data(data_path)

# ---------------- sidebar ----------------
st.sidebar.title("⚙️ Settings")
auto = get_results(df, None, data_path)
# write the cleaned and processed CSVs into data/ the first time the app runs
if not (os.path.exists(CLEANED_PATH) and os.path.exists(PROCESSED_PATH)):
    save_outputs(auto)
rec_k = auto["recommendation"]["recommended_k"]
use_auto = st.sidebar.checkbox(f"Use recommended K ({rec_k})", value=True)
k = rec_k if use_auto else st.sidebar.slider("Number of clusters (K)", 2, 10, rec_k)
res = auto if k == auto["k"] else get_results(df, k, data_path)

cust = res["customers"]
normal_mask = ~cust["IsAnomaly"]
seg_normal = list(cust[normal_mask].groupby("Segment")["Total_Spend"].mean()
                  .sort_values(ascending=False).index)
seg_anom = [g for g in ANOMALY_ORDER if g in set(cust.loc[cust["IsAnomaly"], "Segment"])]
segments = seg_normal + seg_anom
cmap = {s: SEGMENT_COLORS.get(s, "#8D6E63") for s in segments}

st.sidebar.markdown("---")
st.sidebar.caption("Steps: clean data → customer features → IQR outliers (4 anomaly groups) → "
                   "Elbow & Silhouette → K-Means (normal customers)")

# ---------------- header ----------------
st.title("🛒 Customer Segmentation & Anomaly Detection")
st.caption("E-commerce transactions (UK online retailer, Dec 2010 – Dec 2011) · "
           "K-Means clustering · IQR outlier detection")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", f"{len(cust):,}")
c2.metric("Clusters (K)", res["k"])
c3.metric("Silhouette score", f"{res['silhouette']:.3f}")
c4.metric("Anomalies (4 groups)", f"{int(cust['IsAnomaly'].sum()):,}")

tabs = st.tabs(["Customer Features", "Best K (Elbow + Silhouette)", "Clusters",
                "Anomalies", "Customer lookup"])

# ---------------- tab 1: features ----------------
with tabs[0]:
    st.subheader("Features used for clustering (one row per customer)")
    st.markdown("- **Days since last purchase** – how recently the customer bought\n"
                "- **Number of orders** – number of different invoices\n"
                "- **Total spend (£)** – sum of Quantity × UnitPrice")
    st.dataframe(cust[FEATURES].rename(columns=LABELS).describe().round(1), width="stretch")

    cols = st.columns(3)
    for col, f in zip(cols, FEATURES):
        fig = px.histogram(cust[~cust["IsAnomaly"]], x=f, nbins=40,
                           color_discrete_sequence=["#1565C0"], labels=LABELS)
        fig.update_layout(title=LABELS[f], height=280, margin=dict(l=0, r=0, t=40, b=0))
        col.plotly_chart(fig, width="stretch")
    st.caption("Histograms show normal customers only (outliers removed).")

# ---------------- tab 2: K selection ----------------
with tabs[1]:
    sc = res["k_scores"]
    r = res["recommendation"]
    st.subheader("Choosing the number of clusters")
    a, b = st.columns(2)
    fig = px.line(sc, x="K", y="Inertia", markers=True, title="Elbow method (inertia / WCSS)")
    fig.add_vline(x=r["elbow_k"], line_dash="dash", line_color="#D32F2F",
                  annotation_text=f"elbow ≈ {r['elbow_k']}")
    a.plotly_chart(fig, width="stretch")

    fig = px.line(sc, x="K", y="Silhouette", markers=True, title="Silhouette score (higher is better)")
    fig.add_vline(x=r["recommended_k"], line_dash="dash", line_color="#2E7D32",
                  annotation_text=f"chosen K = {r['recommended_k']}")
    b.plotly_chart(fig, width="stretch")

    st.dataframe(sc.style.format({"Inertia": "{:,.0f}", "Silhouette": "{:.3f}"})
                 .highlight_max(subset=["Silhouette"], color="#C8E6C9"),
                 width="stretch", hide_index=True)

    sil_rec = sc.set_index("K").loc[r["recommended_k"], "Silhouette"]
    st.info(
        f"**Result:** the highest silhouette is at **K = {r['best_silhouette_k']}**, but two groups "
        f"are too few to be useful. The elbow curve bends around **K = {r['elbow_k']}**. "
        f"Around the elbow (K = {max(3, r['elbow_k'] - 1)} to {r['elbow_k'] + 1}) the best silhouette "
        f"is at **K = {r['recommended_k']}** (silhouette = {sil_rec:.3f}), so we use "
        f"**K = {r['recommended_k']}**."
    )
    st.caption("Scores are computed on normal customers (outliers removed).")

# ---------------- tab 3: clusters ----------------
with tabs[2]:
    st.subheader("Customer segments")
    summary = (cust.groupby("Segment")
               .agg(Customers=("Total_Spend", "size"),
                    Avg_days_since_last_purchase=("Days_Since_Last_Purchase", "mean"),
                    Avg_orders=("Number_of_Orders", "mean"),
                    Avg_spend=("Total_Spend", "mean"),
                    Total_revenue=("Total_Spend", "sum"))
               .reindex(segments))
    summary["Share_of_customers_%"] = 100 * summary["Customers"] / summary["Customers"].sum()
    summary["Share_of_revenue_%"] = 100 * summary["Total_revenue"] / summary["Total_revenue"].sum()
    summary["Suggested action"] = [action_for(s) for s in summary.index]
    st.dataframe(summary.style.format({
        "Avg_days_since_last_purchase": "{:.0f}", "Avg_orders": "{:.1f}", "Avg_spend": "£{:,.0f}",
        "Total_revenue": "£{:,.0f}", "Share_of_customers_%": "{:.1f}%",
        "Share_of_revenue_%": "{:.1f}%"}), width="stretch")

    a, b = st.columns(2)
    a.plotly_chart(px.pie(summary.reset_index(), names="Segment", values="Customers",
                          color="Segment", color_discrete_map=cmap, hole=0.45,
                          title="Share of customers"), width="stretch")
    b.plotly_chart(px.pie(summary.reset_index(), names="Segment", values="Total_revenue",
                          color="Segment", color_discrete_map=cmap, hole=0.45,
                          title="Share of revenue"), width="stretch")

    normal = cust[~cust["IsAnomaly"]].reset_index()
    st.markdown("#### 3D view of the clusters (normal customers)")
    fig = px.scatter_3d(normal, x="Days_Since_Last_Purchase", y="Number_of_Orders",
                        z="Total_Spend", color="Segment", color_discrete_map=cmap,
                        category_orders={"Segment": seg_normal}, labels=LABELS,
                        opacity=0.7, hover_data=["CustomerID", "Country"], height=650)
    fig.update_traces(marker=dict(size=3))
    st.plotly_chart(fig, width="stretch")

    a, b = st.columns(2)
    a.plotly_chart(px.scatter(normal, x="Number_of_Orders", y="Total_Spend", color="Segment",
                              color_discrete_map=cmap, category_orders={"Segment": seg_normal},
                              labels=LABELS, opacity=0.6, hover_data=["CustomerID"],
                              title="Number of orders vs total spend"), width="stretch")
    b.plotly_chart(px.scatter(normal, x="Days_Since_Last_Purchase", y="Total_Spend",
                              color="Segment", color_discrete_map=cmap,
                              category_orders={"Segment": seg_normal}, labels=LABELS,
                              opacity=0.6, hover_data=["CustomerID"],
                              title="Days since last purchase vs total spend"), width="stretch")

    st.markdown("#### Distribution of each feature by segment")
    feat = st.radio("Feature", FEATURES, format_func=LABELS.get, horizontal=True)
    st.plotly_chart(px.box(normal, x="Segment", y=feat, color="Segment", labels=LABELS,
                           color_discrete_map=cmap, category_orders={"Segment": seg_normal})
                    .update_layout(showlegend=False), width="stretch")

# ---------------- tab 4: anomalies ----------------
with tabs[3]:
    an = cust[cust["IsAnomaly"]].sort_values("Total_Spend", ascending=False)
    st.subheader(f"{len(an)} anomalous customers (outliers)")
    st.markdown("Outliers were found with the **IQR rule**: a customer is an outlier if their "
                "number of orders or total spend is above **Q3 + 1.5 × IQR**. "
                "Values above **Q3 + 3 × IQR** are *extreme* outliers. Anomalies are kept out of "
                "K-Means so they don't pull the cluster centres.")
    st.dataframe(res["iqr_bounds"].rename(index=LABELS)
                 .style.format("{:,.1f}"), width="stretch")

    a, b, c = st.columns(3)
    a.metric("Revenue from anomalies", f"£{an['Total_Spend'].sum():,.0f}")
    b.metric("Share of total revenue",
             f"{100 * an['Total_Spend'].sum() / cust['Total_Spend'].sum():.1f}%")
    c.metric("Avg spend per anomaly", f"£{an['Total_Spend'].mean():,.0f}")

    st.markdown("#### 4 groups of anomalies")
    st.markdown(
        "Each anomaly is grouped by **why** it is an outlier:\n"
        "1. **Extreme bulk buyers** – above the *extreme* limit (3 × IQR) on **both** orders and spend\n"
        "2. **Frequent big spenders** – above the 1.5 × IQR limit on **both** orders and spend\n"
        "3. **Big-order buyers** – outlier on **spend only** (few orders, but very large ones)\n"
        "4. **Very frequent small buyers** – outlier on **orders only** (many orders, normal spend)")

    gsum = (an.groupby("Segment")
            .agg(Customers=("Total_Spend", "size"),
                 Avg_days_since_last_purchase=("Days_Since_Last_Purchase", "mean"),
                 Avg_orders=("Number_of_Orders", "mean"),
                 Avg_spend=("Total_Spend", "mean"),
                 Total_revenue=("Total_Spend", "sum"))
            .reindex(seg_anom))
    gsum["Share_of_total_revenue_%"] = 100 * gsum["Total_revenue"] / cust["Total_Spend"].sum()
    gsum["Suggested action"] = [action_for(s) for s in gsum.index]
    st.dataframe(gsum.style.format({
        "Avg_days_since_last_purchase": "{:.0f}", "Avg_orders": "{:.1f}", "Avg_spend": "£{:,.0f}",
        "Total_revenue": "£{:,.0f}", "Share_of_total_revenue_%": "{:.1f}%"}), width="stretch")

    a, b = st.columns(2)
    a.plotly_chart(px.bar(gsum.reset_index(), x="Customers", y="Segment", orientation="h",
                          color="Segment", color_discrete_map=cmap, text="Customers",
                          category_orders={"Segment": seg_anom},
                          title="Customers in each anomaly group")
                   .update_layout(showlegend=False, yaxis_title=None), width="stretch")
    b.plotly_chart(px.bar(gsum.reset_index(), x="Total_revenue", y="Segment", orientation="h",
                          color="Segment", color_discrete_map=cmap,
                          category_orders={"Segment": seg_anom},
                          labels={"Total_revenue": "Total revenue (£)"},
                          title="Revenue from each anomaly group")
                   .update_layout(showlegend=False, yaxis_title=None), width="stretch")

    plot_df = cust.reset_index()
    plot_df["Group"] = plot_df["Segment"].where(plot_df["IsAnomaly"], "Normal customers")
    gmap = {**{g: cmap[g] for g in seg_anom}, "Normal customers": "#BDBDBD"}
    fig = px.scatter(plot_df, x="Number_of_Orders", y="Total_Spend", color="Group",
                     color_discrete_map=gmap, category_orders={"Group": ["Normal customers"] + seg_anom},
                     log_x=True, log_y=True, opacity=0.75, labels=LABELS,
                     hover_data=["CustomerID", "Days_Since_Last_Purchase"], height=550,
                     title="Anomaly groups vs normal customers (log axes for display only)")
    bounds = res["iqr_bounds"]
    fig.add_hline(y=bounds.loc["Total_Spend", "Upper limit (Q3 + 1.5×IQR)"], line_dash="dash",
                  line_color="#757575", annotation_text="spend limit (1.5×IQR)")
    fig.add_vline(x=bounds.loc["Number_of_Orders", "Upper limit (Q3 + 1.5×IQR)"], line_dash="dash",
                  line_color="#757575", annotation_text="orders limit (1.5×IQR)")
    fig.add_hline(y=bounds.loc["Total_Spend", "Extreme limit (Q3 + 3×IQR)"], line_dash="dot",
                  line_color="#B71C1C", annotation_text="extreme spend (3×IQR)",
                  annotation_position="bottom right")
    fig.add_vline(x=bounds.loc["Number_of_Orders", "Extreme limit (Q3 + 3×IQR)"], line_dash="dot",
                  line_color="#B71C1C", annotation_text="extreme orders (3×IQR)",
                  annotation_position="bottom left")
    st.plotly_chart(fig, width="stretch")
    st.caption("Dashed lines = outlier limits (1.5 × IQR); dotted red lines = extreme limits (3 × IQR).")

    pick = st.selectbox("Show customers in group", ["All anomaly groups"] + seg_anom)
    shown = an if pick == "All anomaly groups" else an[an["Segment"] == pick]
    st.dataframe(shown[["Segment", "Country"] + FEATURES + ["AnomalyReason"]]
                 .rename(columns={**LABELS, "Segment": "Anomaly group"})
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
        st.warning(f"Anomaly reason: {row['AnomalyReason']}")

    tx = res["transactions"]
    st.markdown("Recent transactions")
    st.dataframe(tx[tx["CustomerID"] == cid].sort_values("InvoiceDate", ascending=False).head(50),
                 width="stretch", hide_index=True)

    st.markdown("---")
    seg_filter = st.multiselect("Filter export by segment", segments, default=segments)
    out = cust[cust["Segment"].isin(seg_filter)]
    st.download_button("⬇️ Download customer segments (CSV)", out.to_csv().encode(),
                       "customer_segments.csv", "text/csv")
