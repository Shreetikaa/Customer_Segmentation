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
├── pipeline.py                     # cleaning, features, winsorization, scaling, K-Means, DBSCAN
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

1. **Data cleaning** – remove rows with no CustomerID, cancelled invoices (InvoiceNo starts
   with "C"), quantity/price ≤ 0 and duplicates → 392,692 rows, 4,338 customers.
2. **Customer features** (one row per customer): days since last purchase, number of orders,
   total spend (Quantity × UnitPrice).
3. **Winsorization** – each feature is capped at its 1st and 99th percentile
   (days 1–369, orders 1–30, spend £52–£19,780), so extreme customers cannot distort K-Means.
4. **Normalisation** – MinMaxScaler (default) or StandardScaler, so all 3 features count equally.
5. **K-Means with K = 3** – fixed for interpretability and supported by the data:

   | K | Inertia | Silhouette |
   |---|--------:|-----------:|
   | 2 | 260.0 | 0.569 |
   | **3** | **144.5** | **0.606** |
   | 4 | 110.7 | 0.496 |
   | 5 | 80.3 | 0.486 |
   | 6 | 67.5 | 0.442 |

   The elbow is at K = 3 and K = 3 has the highest silhouette (MinMaxScaler).
6. **DBSCAN anomaly detection** – on the standardised (not winsorized) features,
   min_samples = 6, eps = 0.45 (knee of the k-distance plot). Noise points = anomalies:
   70 customers (1.6%) bringing 34.4% of revenue, grouped by what makes them unusual
   (compared with the 99th percentile).

## Results

| Segment (K-Means) | Customers | Avg days since last purchase | Avg orders | Avg spend | % of revenue |
|---|---:|---:|---:|---:|---:|
| Loyal high-value customers | 277 | 14 | 22.7 | £16,577 | 51.7% |
| Regular customers | 2,999 | 44 | 3.5 | £1,237 | 41.7% |
| Inactive customers | 1,062 | 249 | 1.6 | £552 | 6.6% |

| Anomaly group (DBSCAN) | Customers | Avg orders | Avg spend | % of revenue |
|---|---:|---:|---:|---:|
| Mega buyers (spend and orders above 99th pct) | 21 | 72.4 | £84,979 | 20.1% |
| Big-spend buyers (spend above 99th pct) | 19 | 14.5 | £51,757 | 11.1% |
| Very frequent buyers (orders above 99th pct) | 19 | 43.0 | £10,214 | 2.2% |
| Unusual pattern | 11 | 10.5 | £8,796 | 1.1% |

## App tabs

- **Features & Preprocessing** – the 3 features, winsorization caps and before/after box plot, scaled values
- **Choosing K** – elbow and silhouette charts, why K = 3
- **Clusters** – segment table with actions, customer/revenue share, 3D and 2D scatter plots, box plots
- **Anomalies (DBSCAN)** – eps and min_samples, k-distance plot, the 4 anomaly groups, customer list
- **Customer lookup** – any customer's segment, anomaly flag and transactions, CSV export

The sidebar switches between MinMaxScaler and StandardScaler.
