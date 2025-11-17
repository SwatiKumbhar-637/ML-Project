# main.py

import os
import pandas as pd
import gradio as gr
from modules import association, segmentation, sales_forecasting
from modules import inventory   # ✅ Added only this import

# -----------------------------
# Dataset path
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "stores_sales_forecasting.csv")

# Read dataset
df = pd.read_csv(DATA_PATH, encoding='latin1')
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')

# -----------------------------
# Preprocess for segmentation
# -----------------------------
customer_df, original_df = segmentation.preprocess_segmentation(DATA_PATH)

# -----------------------------
# Create Gradio interface
# -----------------------------
with gr.Blocks(title="Smart Commerce System") as demo:

    gr.Markdown("# Smart Commerce System")

    with gr.Tabs():

        # ----------------------------------------------------
        # 1️⃣ SALES FORECASTING
        # ----------------------------------------------------
        with gr.Tab("Sales Forecasting"):
            gr.Markdown("## Sales Forecasting")

            sales_ui = sales_forecasting.create_sales_ui(
                df,
                num_features=["Sales", "Profit", "Quantity", "Discount"]
            )
            sales_ui

        # ----------------------------------------------------
        # 1.1️⃣ INVENTORY FORECASTING  (✅ ADDED SECTION)
        # ----------------------------------------------------
        with gr.Tab("Inventory Forecasting"):
            gr.Markdown("## Inventory Demand & Stock Forecasting")

            inventory.set_dataframe(df)     # sends df to module
            inv_ui = inventory.build_ui()   # create UI from module
            inv_ui

        # ----------------------------------------------------
        # 2️⃣ SEGMENTATION
        # ----------------------------------------------------
        with gr.Tab("Segmentation"):
            gr.Markdown("## Customer Segmentation and Sales Analytics")
            segmentation_ui = segmentation.combined_ui(customer_df, original_df, df)
            segmentation_ui

        # ----------------------------------------------------
        # 3️⃣ ASSOCIATION RULES
        # ----------------------------------------------------
        with gr.Tab("Association"):
            gr.Markdown("## Association")

            assoc_btn = gr.Button("Show Association")
            assoc_output = gr.HTML()

            def run_association():
                rules = association.generate_rules(df)
                df_display = rules.copy()
                df_display['antecedents'] = df_display['antecedents'].apply(lambda x: ', '.join(list(x)))
                df_display['consequents'] = df_display['consequents'].apply(lambda x: ', '.join(list(x)))
                return df_display.to_html(index=False)

            assoc_btn.click(run_association, outputs=assoc_output)

        # ----------------------------------------------------
        # 4️⃣ PRICE RECOMMENDATION (NEW TAB)
        # ----------------------------------------------------
        with gr.Tab("Price Recommendation"):
            gr.Markdown("## Smart Price Recommendation System")

            price_df = df.copy()

            # -------------------------
            # AGGREGATE CUSTOMER DATA
            # -------------------------
            cust_agg = price_df.groupby("Customer ID")["Sales"].agg(["sum", "mean", "count"])
            cust_agg.columns = ["total_spent", "avg_spent", "num_transactions"]
            cust_agg = cust_agg.reset_index()

            # -------------------------
            # STANDARDIZE FEATURES
            # -------------------------
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import KMeans

            scaler = StandardScaler()
            scaled = scaler.fit_transform(cust_agg[["total_spent", "avg_spent", "num_transactions"]])

            # -------------------------
            # K-MEANS CLUSTERING
            # -------------------------
            kmeans = KMeans(n_clusters=4, random_state=42)
            cust_agg["cluster"] = kmeans.fit_predict(scaled)

            # Cluster mapping
            cluster_labels = {
                0: "High-Value",
                1: "Discount Seekers",
                2: "Bulk Buyers",
                3: "General"
            }

            customers = cust_agg[["Customer ID", "cluster"]].copy()
            customers["cluster_label"] = customers["cluster"].map(cluster_labels)

            # -------------------------
            # FUNCTIONS
            # -------------------------
            def get_customer_details(cid):
                row = customers[customers["Customer ID"] == cid]
                if row.empty:
                    return None
                r = row.iloc[0]
                return {"customer_id": cid, "cluster": r["cluster_label"]}

            def get_base_price(cid, product):
                prev = price_df[(price_df["Customer ID"] == cid) &
                                (price_df["Product Name"] == product)]
                if not prev.empty:
                    return prev.iloc[0]["Sales"]

                prod = price_df[price_df["Product Name"] == product]
                if not prod.empty:
                    return prod["Sales"].mean()

                return None

            def get_adjusted_price(cid, product):
                info = get_customer_details(cid)
                if info is None:
                    return "Customer not found"

                cluster = info["cluster"]
                base_price = get_base_price(cid, product)
                if base_price is None:
                    return "Product not found"

                if cluster == "High-Value":
                    final_price = base_price * 0.95
                elif cluster == "Bulk Buyers":
                    final_price = base_price * 0.92
                elif cluster == "Discount Seekers":
                    final_price = base_price * 0.88
                else:
                    final_price = base_price * 1.05

                return round(final_price, 2)

            def ui_price(cid, product):
                return (
                    get_customer_details(cid),
                    get_base_price(cid, product),
                    get_adjusted_price(cid, product)
                )

            cust_options = sorted(price_df["Customer ID"].unique())
            prod_options = sorted(price_df["Product Name"].unique())

            gr.Interface(
                fn=ui_price,
                inputs=[
                    gr.Dropdown(label="Select Customer", choices=cust_options),
                    gr.Dropdown(label="Select Product", choices=prod_options)
                ],
                outputs=[
                    gr.JSON(label="Customer Details"),
                    gr.Number(label="Base Price"),
                    gr.Number(label="Recommended Price")
                ],
                allow_flagging="never"
            ).render()

# -----------------------------
# Launch the app
# -----------------------------
if __name__ == "__main__":
    demo.launch()
