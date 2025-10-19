# Customer Segmentation & Churn Analysis
# Author: Rudranil Mallik
# Dataset: Online Retail (Kaggle)

import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Step 1: Load Dataset
# -----------------------------
data = pd.read_csv(r"E:\Project_New\Customer\OnlineRetail.csv", encoding='ISO-8859-1') # Replace with your dataset path
print("Dataset Loaded:")
print(data.head())
print(data.info())

# -----------------------------
# Step 2: Data Cleaning
# -----------------------------
# Remove missing CustomerID
data = data[data['CustomerID'].notnull()]

# Remove negative or zero quantities
data = data[data['Quantity'] > 0]

# Drop duplicates
data = data.drop_duplicates()

# Create TotalPrice column
data['TotalPrice'] = data['Quantity'] * data['UnitPrice']

# -----------------------------
# Step 3: Feature Engineering (RFM Analysis)
# -----------------------------
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], dayfirst=True)
snapshot_date = data['InvoiceDate'].max() + pd.Timedelta(days=1)

rfm = data.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # Recency
    'InvoiceNo': 'count',                                    # Frequency
    'TotalPrice': 'sum'                                      # Monetary
})

rfm.rename(columns={
    'InvoiceDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'TotalPrice': 'Monetary'
}, inplace=True)

print("\nRFM Table:")
print(rfm.head())

# -----------------------------
# Step 4: Customer Segmentation (K-Means)
# -----------------------------
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

print("\nCustomer Segments:")
print(rfm['Cluster'].value_counts())

# Visualization of clusters
sns.countplot(x='Cluster', data=rfm)
plt.title("Customer Segments Distribution")
plt.show()

# -----------------------------
# Step 5: Churn Definition
# -----------------------------
# Define churn: customers who haven't purchased in last 90 days
rfm['Churn'] = rfm['Recency'].apply(lambda x: 1 if x > 90 else 0)
sns.countplot(x='Churn', data=rfm)
plt.title("Churn Distribution")
plt.show()

# -----------------------------
# Step 6: Churn Prediction
# -----------------------------
X = rfm[['Recency', 'Frequency', 'Monetary']]
y = rfm['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nChurn Prediction Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# Step 7: Optional - Insights
# -----------------------------
# Average Recency, Frequency, Monetary by Cluster
cluster_summary = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
print("\nCluster Summary:")
print(cluster_summary)

# End of Script
