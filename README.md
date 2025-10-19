# Customer Segmentation & Churn Analysis

## Overview
This project performs **customer segmentation** and **churn prediction** for an e-commerce business using the **Online Retail dataset**. The goal is to identify high-value customer segments, understand purchase behavior, and predict customers who are likely to churn.  

This project demonstrates **data cleaning, feature engineering, clustering, predictive modeling, and visualization**, which are key skills for a Data Analyst role.

---

## Features
- **RFM Analysis**: Calculates **Recency, Frequency, and Monetary** values for each customer.
- **Customer Segmentation**: Uses **K-Means clustering** to group customers into meaningful segments.
- **Churn Prediction**: Identifies customers at risk of churn (no purchase in last 90 days) using **Random Forest**.
- **Data Visualization**: Visualizes customer segments and churn distribution using **Seaborn and Matplotlib**.
- **Insights**: Summarizes average Recency, Frequency, and Monetary values by cluster for business decisions.

---

## Dataset
- Source: [Online Retail Dataset (Kaggle)](https://www.kaggle.com/datasets/hellbuoy/online-retail-dataset)
- Key columns:
  - `InvoiceNo` – Invoice number
  - `StockCode` – Product code
  - `Description` – Product description
  - `Quantity` – Quantity purchased
  - `InvoiceDate` – Invoice date and time
  - `UnitPrice` – Price per unit
  - `CustomerID` – Unique customer identifier
  - `Country` – Customer country
