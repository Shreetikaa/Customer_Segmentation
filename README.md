# Customer Segmentation & Anomaly Detection (E-Commerce Data)

Dataset: Kaggle – [carrie1/ecommerce-data](https://www.kaggle.com/datasets/carrie1/ecommerce-data)
(UK online retailer, Dec 2010 – Dec 2011, 541,909 rows).

## Project structure

```
customer_segmentation/
├── data/
│   ├── online_retail.csv           # raw Kaggle dataset (541,909 rows)
│   ├── online_retail_cleaned.csv   # cleaned transactions (392,692 rows)
│   └── customer_segments.csv       # processed: one row per customer + segment (4,338)
├── pipeline.py                     # cleaning, features, IQR anomalies, elbow/silhouette, K-Means
├── app.py                          # Streamlit dashboard
├── requirements.txt
└── README.md
```

## How to run

```
pip3 install -r requirements.txt
python3 pipeline.py        # creates the cleaned and processed CSVs in data/
streamlit run app.py       # opens the dashboard
```
(The app also creates the cleaned and processed CSVs automatically if they are missing.)

## Method

1. **Data cleaning (data quality)** – remove rows with no CustomerID, cancelled invoices
   (InvoiceNo starts with "C"), quantity/price ≤ 0 and duplicates → 4,338 customers.
2. **Customer features** (one row per customer)
   - Days since last purchase
   - Number of orders (unique invoices)
   - Total spend (Quantity × UnitPrice)
3. **Outliers / anomalies – IQR rule** on number of orders and total spend:
   outlier if value > Q3 + 1.5 × IQR (or < Q1 − 1.5 × IQR).
   - Orders: upper limit 11 · Spend: upper limit £3,692
   - 474 customers flagged → not used in K-Means.
   - Anomalies are split into **4 groups** by *why* they are outliers
     (extreme limit = Q3 + 3 × IQR → 17 orders / £5,723):
     1. Extreme bulk buyers – extreme on both orders and spend
     2. Frequent big spenders – outlier on both
     3. Big-order buyers – outlier on spend only
     4. Very frequent small buyers – outlier on orders only
4. **Choosing K** – K-Means for K = 2…10 on the normal customers:

   | K | Inertia | Silhouette |
   |---|--------:|-----------:|
   | 2 | 714.8M | **0.681** |
   | 3 | 346.9M | **0.604** |
   | 4 | 210.9M | 0.540 |
   | 5 | 148.9M | 0.505 |
   | 6 | 114.5M | 0.492 |
   | 7 | 93.1M | 0.457 |
   | 8 | 76.8M | 0.434 |
   | 9 | 65.3M | 0.425 |
   | 10 | 55.3M | 0.427 |

   K=2 has the highest silhouette but two groups are too few. The elbow is at K≈4;
   around the elbow (K=3–5) the best silhouette is K=3 → **K = 3**.
5. **K-Means** with K = 3; clusters named by average spend.

## Result (K = 3)

| Segment | Customers | Avg days since last purchase | Avg orders | Avg spend |
|---|---:|---:|---:|---:|
| High spenders | 433 | 44 | 6.0 | £2,666 |
| Medium spenders | 971 | 61 | 4.0 | £1,301 |
| Low spenders | 2,460 | 127 | 1.7 | £370 |
| Anomaly – Extreme bulk buyers | 105 | 7 | 36.0 | £28,594 |
| Anomaly – Frequent big spenders | 131 | 19 | 15.6 | £7,665 |
| Anomaly – Big-order buyers | 189 | 39 | 7.0 | £7,523 |
| Anomaly – Very frequent small buyers | 49 | 21 | 14.6 | £2,676 |

## App tabs

- **Customer Features** – the 3 features and their distributions
- **Best K** – elbow curve, silhouette curve, table and explanation
- **Clusters** – segment summary with suggested actions, customer/revenue share,
  3D and 2D scatter plots, box plots
- **Anomalies** – IQR limits, the 4 anomaly groups (table, bar charts, scatter with limit lines), customer list per group
- **Customer lookup** – any customer's segment and transactions, CSV export

The sidebar lets you change K; everything updates live.
