import gradio as gr
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# --------------------------------------------------------
# 🔹 LOAD OR RECEIVE DATAFRAME (Expect df to be imported)
# --------------------------------------------------------
df = None  # This will be injected from main.py

def set_dataframe(dataframe):
    global df
    df = dataframe.copy()
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")


# --------------------------------------------------------
# 🔹 1. MONTHLY DEMAND
# --------------------------------------------------------
def get_monthly_demand(sub_category):
    sub_df = df[df["Sub-Category"] == sub_category].copy()
    if sub_df.empty:
        return pd.Series(dtype="float")

    monthly = (
        sub_df.set_index("Order Date")["Quantity"]
        .resample("ME")
        .sum()
        .fillna(0)
        .sort_index()
    )
    return monthly


# --------------------------------------------------------
# 🔹 2. FEATURE ENGINEERING
# --------------------------------------------------------
def build_training_data(monthly):

    df_model = pd.DataFrame({
        "Quantity": monthly,
        "Lag1": monthly.shift(1),
        "Lag2": monthly.shift(2),
        "Lag3": monthly.shift(3),
        "Lag6": monthly.shift(6),
    })

    df_model["Rolling3"] = monthly.rolling(3).mean().fillna(0)
    df_model["Rolling6"] = monthly.rolling(6).mean().fillna(0)

    df_model["Month"] = df_model.index.month
    df_model["Quarter"] = df_model.index.quarter
    df_model["Year"] = df_model.index.year

    return df_model.fillna(0)


# --------------------------------------------------------
# 🔹 3. FORECAST + INVENTORY + METRICS
# --------------------------------------------------------
def evaluate_and_forecast(subcat):

    monthly = get_monthly_demand(subcat)

    if len(monthly) < 12:
        return (
            f"Not enough data for {subcat}",
            None,
            monthly.reset_index(),
            "N/A", "N/A", "N/A", "N/A"
        )

    df_model = build_training_data(monthly)

    X = df_model.drop("Quantity", axis=1)
    y = df_model["Quantity"]

    test_size = 6
    train_size = len(df_model) - test_size

    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=12,
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    # --------------------------- METRICS -----------------------------
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    smape = np.mean(
        2 * np.abs(y_test - preds) /
        (np.abs(y_test) + np.abs(preds) + 1e-6)
    ) * 100
    accuracy = 100 - smape

    # --------------------------- INVENTORY ---------------------------
    last_X = X.iloc[-1].values.reshape(1, -1)
    forecast_next_month = model.predict(last_X)[0]

    safety_stock = 1.65 * monthly[-6:].std()

    reorder_point = forecast_next_month + safety_stock
    recommended_order_qty = max(0, round(reorder_point))

    # --------------------------- TOTAL DEMAND -------------------------
    total_demand = monthly.sum()

    # --------------------------- PLOT ------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(y_test.index, y_test.values, label="Actual", marker="o")
    ax.plot(y_test.index, preds, label="Predicted", marker="o")
    ax.legend()
    ax.set_title(f"{subcat} - Actual vs Predicted")
    ax.grid()

    metrics_text = (
        f"Sub-Category: {subcat}\n"
        f"MAE: {mae:.2f}\n"
        f"RMSE: {rmse:.2f}\n"
        f"sMAPE: {smape:.2f}%\n"
        f"Accuracy: {accuracy:.2f}%\n"
    )

    return (
        metrics_text,
        fig,
        monthly.reset_index(),
        total_demand,
        round(forecast_next_month, 2),
        round(safety_stock, 2),
        round(reorder_point, 2),
        recommended_order_qty
    )


# --------------------------------------------------------
# 🔹 4. BUILD GRADIO UI
# --------------------------------------------------------
def build_ui():
    subcats = sorted(df["Sub-Category"].unique())

    ui = gr.Interface(
        fn=evaluate_and_forecast,
        inputs=gr.Dropdown(subcats, label="Select Sub-Category"),
        outputs=[
            gr.Textbox(label="Forecasting Metrics"),
            gr.Plot(label="Actual vs Predicted Plot"),
            gr.Dataframe(label="Monthly Demand"),
            gr.Number(label="Total Demand"),

            gr.Number(label="Next Month Forecast"),
            gr.Number(label="Safety Stock"),
            gr.Number(label="Reorder Point"),
            gr.Number(label="Recommended Order Quantity")
        ],
        title="Inventory Forecasting System",
        description="Forecasts demand, shows accuracy, and calculates safety stock & reorder point."
    )
    return ui


# expose UI object for use in main.py
ui = None

def launch_ui():
    global ui
    ui = build_ui()
    ui.launch(debug=True)

