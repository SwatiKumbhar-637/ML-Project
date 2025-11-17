Smart Commerce System
=====================

Overview
--------
The Smart Commerce System is a unified analytics platform designed to support sales forecasting, customer segmentation, association rule mining, and inventory planning. It provides an interactive interface built with Gradio and integrates multiple machine learning and statistical models to generate insights for retail and e-commerce businesses.

Features
--------

1. Sales Forecasting
   - Predicts future sales using machine learning and regression models.
   - Supports filtering by category, region, segment, and product hierarchy.
   - Generates trend charts and exportable prediction tables.

2. Customer Segmentation
   - Performs RFM-based segmentation.
   - Provides customer behavior analysis and visual summaries.

3. Association Rule Mining
   - Identifies co-purchased products using Apriori.
   - Generates actionable rules for cross-selling and product bundling.

4. Inventory Forecasting
   - Computes forecasted demand, safety stock, and reorder point.
   - Supports category and sub-category level forecasting.

5. Advanced Region-Based Forecasting
   - Compares multiple time-series models:
       - Exponential Smoothing
       - ARIMA
       - Moving Average
       - Naive Model
       - Linear Regression with lag features
   - Selects models based on MAE and generates a 6-month forecast.

Technology Used
---------------
- Python
- Pandas, NumPy
- Scikit-learn
- Statsmodels
- MLxtend
- Gradio
- Matplotlib

Project Structure
-----------------
SalesDashboardProject/
│
├── main.py
├── modules/
│   ├── sales_forecasting.py
│   ├── segmentation.py
│   ├── association.py
│   ├── inventory.py
│   ├── advanced_forecasting.py
│
├── data/
│   └── stores_sales_forecasting.csv
│
└── README.txt

How to Run
----------
1. Install dependencies:
   pip install -r requirements.txt

2. Start the application:
   python main.py

3. The Gradio interface will open in your browser.

Usage
-----
Select the desired module from the tabbed interface and explore insights, forecasts, or rules using the uploaded dataset.
