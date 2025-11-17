import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from scipy.stats.mstats import winsorize


def create_sales_ui(df, num_features=None):
    """
    Create Gradio UI for Sales Forecasting.
    df: pandas DataFrame (already loaded)
    num_features: list of numerical features for winsorization
    """

    # -------------------- Outlier Winsorization --------------------
    if num_features:
        for feature in num_features:
            df[feature] = winsorize(df[feature], limits=[0.05, 0.05])
            print(f"Outliers capped for {feature}")

    df["Order Date"] = pd.to_datetime(df["Order Date"])

    # -------------------- Monthly aggregation --------------------
    monthly = df.groupby(df["Order Date"].dt.to_period("M"))["Sales"].sum()
    monthly = monthly.to_timestamp().reset_index()
    monthly.columns = ["Month", "Sales"]
    monthly = monthly.sort_values("Month").reset_index(drop=True)
    mean_sales = monthly["Sales"].mean()  # For percentage calculations

    # -------------------- Create lag features --------------------
    monthly["Lag1"] = monthly["Sales"].shift(1)
    monthly["Lag2"] = monthly["Sales"].shift(2)
    monthly["Lag3"] = monthly["Sales"].shift(3)

    # -------------------- Add time features --------------------
    monthly["Month_Num"] = monthly["Month"].dt.month
    monthly["Quarter"] = monthly["Month"].dt.quarter
    monthly["Year"] = monthly["Month"].dt.year
    monthly = monthly.dropna().reset_index(drop=True)

    # -------------------- 80/20 Split --------------------
    split_idx = int(len(monthly) * 0.80)
    train = monthly.iloc[:split_idx]
    test = monthly.iloc[split_idx:]

    features = ["Lag1", "Lag2", "Lag3", "Month_Num", "Quarter", "Year"]
    X_train, y_train = train[features], train["Sales"]
    X_test, y_test = test[features], test["Sales"]

    # -------------------- Train models --------------------
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "XGBoost": XGBRegressor(objective="reg:squarederror", n_estimators=300,
                                 learning_rate=0.1, max_depth=4, random_state=42)
    }

    for m in models:
        models[m].fit(X_train, y_train)

    # -------------------- Helper Functions --------------------
    def evaluate_model(selected_model):
        model = models[selected_model]
        y_pred = model.predict(X_test)

        # ⭐ Round predictions to 2 decimals
        y_pred = np.round(y_pred, 2)

        results = pd.DataFrame({
            "Month": test["Month"].dt.strftime("%Y-%m"),
            "Actual Sales": y_test.values,
            "Predicted Sales": y_pred
        })

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        accuracy_pct = round(r2*100, 2)
        mae_pct = round(mae / mean_sales * 100, 2)
        rmse_pct = round(rmse / mean_sales * 100, 2)

        metrics_df = pd.DataFrame({
            "Metric": ["MAE", "RMSE", "R²", "Accuracy (%)", "MAE (%)", "RMSE (%)"],
            "Value": [round(mae,2), round(rmse,2), round(r2,4), accuracy_pct, mae_pct, rmse_pct]
        })

        # Plot
        plt.figure(figsize=(12,6))
        plt.plot(results["Month"], results["Actual Sales"], marker="o", label="Actual")
        plt.plot(results["Month"], results["Predicted Sales"], marker="x", label="Predicted")
        plt.xticks(rotation=45)
        plt.xlabel("Month")
        plt.ylabel("Sales")
        plt.title(f"{selected_model} - Actual vs Predicted Sales")
        plt.legend()
        plt.tight_layout()
        fig = plt.gcf()
        return results, metrics_df, fig

    def show_overview():
        plt.figure(figsize=(12,5))
        plt.plot(monthly["Month"], monthly["Sales"], marker="o", color="tab:blue")
        plt.xlabel("Month")
        plt.ylabel("Sales")
        plt.title("Monthly Sales Overview")
        plt.xticks(rotation=45)
        plt.tight_layout()
        return plt.gcf()

    def comparison_table():
        data = []
        for name, model in models.items():
            y_pred = model.predict(X_test)

            # ⭐ Round predictions here also if shown later
            y_pred = np.round(y_pred, 2)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            acc = round(r2*100,2)
            mae_pct = round(mae / mean_sales * 100, 2)
            rmse_pct = round(rmse / mean_sales * 100, 2)
            data.append([name, round(mae,2), mae_pct, round(rmse,2), rmse_pct, round(r2,4), acc])

        return pd.DataFrame(data, columns=["Model","MAE","MAE (%)","RMSE","RMSE (%)","R²","Accuracy (%)"])

    # -------------------- Build Gradio UI --------------------
    with gr.Blocks() as sales_ui:
        gr.Markdown("<h2>📊 Monthly Sales Forecasting Dashboard</h2>")
        with gr.Tabs():
            with gr.Tab("Overview"):
                gr.Plot(show_overview, label="Monthly Sales Overview")
            with gr.Tab("Model Evaluation"):
                model_dropdown = gr.Dropdown(list(models.keys()), label="Select Model")
                run_btn = gr.Button("Run Evaluation")
                results_out = gr.Dataframe(label="Actual vs Predicted")
                metrics_out = gr.Dataframe(label="Model Metrics")
                plot_out = gr.Plot(label="Graph: Actual vs Predicted")
                run_btn.click(
                    evaluate_model,
                    inputs=[model_dropdown],
                    outputs=[results_out, metrics_out, plot_out]
                )
            with gr.Tab("Model Comparison"):
                comparison_out = gr.Dataframe(value=comparison_table(), label="All Models Comparison")
    return sales_ui
