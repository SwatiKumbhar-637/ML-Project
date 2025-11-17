# modules/segmentation.py

# ----------------------------- Imports -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import gradio as gr

# ----------------------------- Data Preprocessing -----------------------------
def preprocess_segmentation(csv_path):
    """Load data, scale features, and create clusters"""
    original_df = pd.read_csv(csv_path, encoding='latin1')
    features = original_df[['Customer ID', 'Sales', 'Profit', 'Quantity', 'Discount']]
    customer_df = features.groupby('Customer ID').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Quantity': 'sum',
        'Discount': 'mean'
    }).reset_index()

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(customer_df[['Sales','Profit','Quantity','Discount']])

    # KMeans clustering
    k = 4
    kmeans = KMeans(n_clusters=k, random_state=42)
    customer_df['Cluster'] = kmeans.fit_predict(scaled_data)

    # Assign meaningful cluster names
    cluster_profile = customer_df.groupby("Cluster")[['Sales','Profit','Quantity','Discount']].mean()
    high_sales_cluster = cluster_profile["Sales"].idxmax()
    low_sales_cluster = cluster_profile["Sales"].idxmin()
    high_discount_cluster = cluster_profile["Discount"].idxmax()
    high_quantity_cluster = cluster_profile["Quantity"].idxmax()

    cluster_name_map = {}
    for c in cluster_profile.index:
        if c == high_sales_cluster:
            cluster_name_map[c] = "High-Value Customers"
        elif c == high_discount_cluster:
            cluster_name_map[c] = "Discount Seekers"
        elif c == high_quantity_cluster:
            cluster_name_map[c] = "Bulk Buyers"
        elif c == low_sales_cluster:
            cluster_name_map[c] = "Low-Value Customers"
        else:
            cluster_name_map[c] = "General Customers"

    customer_df["Cluster_Name"] = customer_df["Cluster"].map(cluster_name_map)

    return customer_df, original_df

# ----------------------------- Customer Segmentation UI -----------------------------
def run_customer_segmentation_ui(customer_df, original_df):
    # Merge Customer Name from original_df
    customer_name_map = original_df[['Customer ID', 'Customer Name']].drop_duplicates(subset=['Customer ID'])
    if 'Customer Name' in customer_df.columns:
        customer_df = customer_df.drop(columns=['Customer Name'])
    customer_df = customer_df.merge(customer_name_map, on='Customer ID', how='left')

    # First Dashboard (same plotting code)
    def show_segmentation_ui():
        plots = []

        cluster_counts = customer_df["Cluster_Name"].value_counts()
        fig1 = plt.figure(figsize=(6,4))
        sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette="Set2")
        plt.title("Customer Count per Cluster")
        plt.xlabel("Cluster Type")
        plt.ylabel("Number of Customers")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plots.append(fig1)

        cluster_profile = customer_df.groupby("Cluster")[['Sales','Profit','Quantity','Discount']].mean()
        fig2 = plt.figure(figsize=(8,5))
        sns.heatmap(cluster_profile, annot=True, cmap="YlGnBu", fmt=".2f")
        plt.title("Cluster Profile Heatmap")
        plt.tight_layout()
        plots.append(fig2)

        customer_table = customer_df[['Customer ID', 'Customer Name',
                                      'Sales','Profit','Quantity','Discount','Cluster_Name']].head(20)
        customer_table[['Sales','Profit','Quantity','Discount']] = customer_table[['Sales','Profit','Quantity','Discount']].round(2)
        return customer_table, *plots

    # Filtered Customers — UPDATED (Heatmap Removed)
    def filter_customers(cluster_type):
        filtered = customer_df[customer_df['Cluster_Name'] == cluster_type].sort_values(by='Sales', ascending=False)
        customer_table = filtered.head(20)[['Customer ID', 'Customer Name',
                                            'Sales','Profit','Quantity','Discount','Cluster_Name']]
        customer_table[['Sales','Profit','Quantity','Discount']] = customer_table[['Sales','Profit','Quantity','Discount']].round(2)

        plots = []

        top_customers = filtered.head(20)
        fig1 = plt.figure(figsize=(8,4))
        sns.barplot(x='Customer ID', y='Sales', data=top_customers, palette="Blues_d")
        plt.title(f"Top 20 Customers in {cluster_type} Cluster (Sales)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plots.append(fig1)

        # Heatmap removed

        return customer_table, *plots

    # ----------------------------- GRADIO UI -----------------------------
    with gr.Column() as customer_ui:
        with gr.Tabs():
            # ---------- Overall Segmentation ----------
            with gr.TabItem("Overall Segmentation"):
                gr.Markdown("## Customer Segmentation Dashboard")
                table1 = gr.Dataframe(label="Customer Data Sample")
                plot1 = gr.Plot(label="Customer Count per Cluster")
                plot2 = gr.Plot(label="Cluster Profile Heatmap")
                refresh_btn = gr.Button("Refresh Data")

                def update_first():
                    table, p1, p2 = show_segmentation_ui()
                    return table, p1, p2

                refresh_btn.click(update_first, outputs=[table1, plot1, plot2])

            # ---------- Filter by Cluster ----------
            with gr.TabItem("Filter by Cluster"):
                gr.Markdown("## Filter Customers by Cluster")

                cluster_dropdown = gr.Dropdown(
                    choices=customer_df['Cluster_Name'].unique().tolist(),
                    label="Select Cluster"
                )

                table2 = gr.Dataframe(label="Filtered Customers")
                plot3 = gr.Plot(label="Top 20 Customers Sales")

                filter_btn = gr.Button("Show Filtered Data")

                filter_btn.click(
                    filter_customers,
                    inputs=cluster_dropdown,
                    outputs=[table2, plot3]   # Heatmap removed
                )

    return customer_ui

# ----------------------------- Sales Analytics UI -----------------------------
def run_sales_analytics_ui(df):
    sns.set_theme(style="whitegrid")

    # Analysis Functions
    def sales_trend_monthly():
        df["Month"] = df["Order Date"].dt.strftime("%b")
        df["MonthNum"] = df["Order Date"].dt.month
        monthly_sales = df.groupby(["MonthNum","Month"])["Sales"].sum().sort_index()
        plt.figure(figsize=(10,5))
        plt.plot(monthly_sales.index.get_level_values("Month"), monthly_sales.values, color="#1f77b4", linewidth=2)
        plt.title("Monthly Sales Trend", fontsize=14)
        plt.xlabel("Month")
        plt.ylabel("Total Sales")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()

    def yearly_sales_analysis():
        df["Year"] = df["Order Date"].dt.year
        yearly_sales = df.groupby("Year")["Sales"].sum()
        plt.figure(figsize=(10,5))
        plt.bar(yearly_sales.index, yearly_sales.values, color="#1f77b4")
        plt.title("Yearly Sales Analysis", fontsize=14)
        plt.xlabel("Year")
        plt.ylabel("Total Sales")
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        return plt.gcf()

    def subcategory_sales():
        plt.figure(figsize=(10,5))
        order = df.groupby("Sub-Category")["Sales"].sum().sort_values()
        sns.barplot(x=order.values, y=order.index, palette="viridis")
        plt.title("Sales by Sub-Category", fontsize=14)
        plt.xlabel("Total Sales")
        plt.ylabel("Sub-Category")
        plt.tight_layout()
        return plt.gcf()

    def region_sales():
        plt.figure(figsize=(10,7))
        order = df.groupby("State")["Sales"].sum().sort_values(ascending=False).head(20)
        sns.barplot(y=order.index, x=order.values, palette="viridis")
        plt.title("Top 20 States by Sales", fontsize=14)
        plt.xlabel("Total Sales")
        plt.ylabel("State")
        plt.tight_layout()
        return plt.gcf()

    def discount_profit():
        plt.figure(figsize=(8,5))
        sns.scatterplot(x=df["Discount"], y=df["Profit"], color="#1f77b4", alpha=0.6)
        plt.title("Discount vs Profit", fontsize=14)
        plt.xlabel("Discount")
        plt.ylabel("Profit")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()

    def shipping_mode_analysis():
        plt.figure(figsize=(8,5))
        ship_sales = df.groupby("Ship Mode")["Sales"].sum().sort_values()
        sns.barplot(x=ship_sales.values, y=ship_sales.index, palette=["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"])
        plt.title("Sales by Shipping Mode", fontsize=14)
        plt.xlabel("Total Sales")
        plt.ylabel("Shipping Mode")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()

    def quantity_distribution():
        plt.figure(figsize=(10,5))
        sns.histplot(df["Quantity"], kde=True, color="#1f77b4")
        plt.title("Distribution of Quantity Sold", fontsize=14)
        plt.xlabel("Quantity Purchased per Order")
        plt.ylabel("Number of Orders")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()

    def correlation_heatmap():
        plt.figure(figsize=(8,6))
        corr = df[["Sales","Profit","Quantity","Discount"]].corr()
        sns.heatmap(corr, annot=True, cmap="Blues", linewidths=0.5)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        return plt.gcf()

    def dashboard():
        return (
            sales_trend_monthly(),
            yearly_sales_analysis(),
            subcategory_sales(),
            shipping_mode_analysis(),
            region_sales(),
            discount_profit(),
            quantity_distribution(),
            correlation_heatmap(),
        )

    # Gradio UI Layout
    with gr.Column() as analytics_ui:
        with gr.Row():
            st = gr.Plot(label="Monthly Sales Trend")
            pt = gr.Plot(label="Yearly Sales Analysis")
        with gr.Row():
            sc = gr.Plot(label="Sales by Sub-Category")
            sm = gr.Plot(label="Shipping Mode Sales")
        with gr.Row():
            rs = gr.Plot(label="State-wise Sales")
            dp = gr.Plot(label="Discount vs Profit")
        with gr.Row():
            qd = gr.Plot(label="Quantity Distribution")
            hm = gr.Plot(label="Correlation Heatmap")

        btn = gr.Button("Generate Dashboard", variant="primary")
        btn.click(fn=dashboard, inputs=None, outputs=[st, pt, sc, sm, rs, dp, qd, hm])

    return analytics_ui

# ----------------------------- Combined UI -----------------------------
def combined_ui(customer_df, original_df, df):
    # Return a single UI component that contains both segmentation and analytics tabs
    with gr.Column() as ui:
        with gr.Tabs():
            with gr.TabItem("Customer Segmentation"):
                run_customer_segmentation_ui(customer_df, original_df)
            with gr.TabItem("Sales Analytics"):
                run_sales_analytics_ui(df)
    return ui
