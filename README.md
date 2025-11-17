Smart Commerce System
The Smart Commerce System is an integrated retail analytics platform that provides Sales Forecasting, Inventory Forecasting, Customer Segmentation, Association Rule Mining, and Price Recommendation.
It is built using Python, Gradio, and a collection of statistical and machine-learning techniques.

1. Project Overview
This system enables retail businesses to make data-driven decisions through:

1.1 Sales Forecasting
Predicts product-level sales using Random Forest Regression

Includes outlier handling (Winsorization) and feature engineering

Reports model metrics such as MAE, RMSE, and R²

1.2 Inventory Forecasting
Forecasts SKU and sub-category demand

Computes:

Expected monthly demand

Demand variability

Safety Stock

Reorder Point

Recommended inventory levels

Based on standard supply-chain formulas

1.3 Customer Segmentation
K-Means clustering on customer spending behaviour

Identifies groups such as high-value customers, bulk buyers, and discount-sensitive customers

Allows customer-level analytics

1.4 Association Rule Mining
Market Basket Analysis using Apriori

Generates rules with support, confidence, and lift

Useful for product bundling and cross-sell recommendations

1.5 Price Recommendation
Customer-specific dynamic pricing

Uses segmentation output to adjust pricing strategies

Computes base price and recommended price per customer-product pair

2. Technology Stack
Component	Technology
Interface	Gradio
Machine Learning	Scikit-Learn, Statsmodels
Clustering	K-Means
Association Rules	MLxtend Apriori
Data Processing	Pandas, NumPy
Visualizations	Matplotlib
3. Project Structure
smart-commerce-system/
│
├── data/
│   └── stores_sales_forecasting.csv
│
├── modules/
│   ├── sales_forecasting.py
│   ├── inventory.py
│   ├── segmentation.py
│   └── association.py
│
├── main.py
├── README.md
└── requirements.txt
4. Installation and Setup
Step 1: Clone the repository
git clone https://github.com/yourusername/smart-commerce-system.git
cd smart-commerce-system
Step 2: Install dependencies
pip install -r requirements.txt
Step 3: Run the application
python main.py
Step 4
Open the Gradio link (usually http://127.0.0.1:7860) in a browser.

5. Feature Details
5.1 Sales Forecasting
Uses Random Forest to predict sales based on features

Handles skewed numerical features

Provides predictions and error metrics

5.2 Inventory Forecasting
Safety Stock Formula

Safety Stock = Z * σ_demand * √(Lead Time)
Reorder Point Formula

Reorder Point = (Average Demand × Lead Time) + Safety Stock
5.3 Customer Segmentation
Standardization of customer metrics

K-Means clustering into spending-based groups

Supports targeting, promotions, and pricing

5.4 Association Rules
Generates the strongest rules using Apriori

Helps identify product relationships

5.5 Price Recommendation
Combines segmentation + sales history

Adjusts prices depending on customer behaviour patterns

Produces base and recommended price

6. Possible Future Enhancements
Deep Learning forecasting models (LSTM, Prophet)

Economic Order Quantity (EOQ)

Collaborative filtering for recommendations

Automated dashboard/reporting

Deployment on cloud platforms

8. Contributors
Swati Kumbhar 
Shrutika Nalavade
Radhika Kulakarni
