# =====================================================================
# ADVANCED FORECASTING MODULE (REGION-BASED + MODEL COMPARISON)
# =====================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

import gradio as gr


# =====================================================================
# 1. LOAD & CLEAN DATA
# =====================================================================

def preprocess(df):

    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df = df.dropna(subset=["Order Date"])

    df["Order_YearMonth"] = df["Order Date"].dt.to_period("M")
    df["Order_Year"] = df["Order Date"].dt.year
    df["Order_Month"] = df["Order Date"].dt.month
    df["Order_Quarter"] = df["Order Date"].dt.quarter

    return df


# =====================================================================
# 2. CREATE DEMOGRAPHIC TIME SERIES
# =====================================================================

def create_demographic_ts(df, groups):
    ts = df.groupby(["Order_YearMonth"] + groups).agg(
        Sales="sum",
        Quantity="sum",
        Profit="sum",
        Order_Count=("Order ID", "nunique"),
    ).reset_index()
    ts["Order_YearMonth"] = ts["Order_YearMonth"].astype(str)
    return ts


# =====================================================================
# 3. PREPARE REGION-SPECIFIC SERIES
# =====================================================================

def prepare_region_ts(region_ts, region_name):
    data = region_ts[region_ts["Region"] == region_name].copy()
    data = data.sort_values("Order_YearMonth")
    data.set_index("Order_YearMonth", inplace=True)
    return data["Sales"]


# =====================================================================
# 4. MODEL COMPARISON
# =====================================================================

def create_features(series):
    features = []
    for i in range(len(series)):
        if i >= 12:
            features.append({
                "lag1": series[i - 1],
                "lag2": series[i - 2],
                "lag3": series[i - 3],
                "lag12": series[i - 12],
                "roll_mean_3": np.mean(series[i - 3:i]),
                "roll_std_3": np.std(series[i - 3:i])
            })
    return pd.DataFrame(features)


def compare_models(series):
    if len(series) < 24:
        return None

    split = int(len(series) * 0.8)
    train, test = series[:split], series[split:]

    results = {}

    # Exponential Smoothing
    try:
        es_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12)
        es_fit = es_model.fit()
        es_pred = es_fit.forecast(len(test))
        results["Exponential_Smoothing"] = mean_absolute_error(test, es_pred)
    except:
        results["Exponential_Smoothing"] = np.nan

    # ARIMA
    try:
        arima = ARIMA(train, order=(1, 1, 1)).fit()
        arima_pred = arima.forecast(len(test))
        results["ARIMA"] = mean_absolute_error(test, arima_pred)
    except:
        results["ARIMA"] = np.nan

    # Moving Average
    try:
        sma_pred = np.array([train[-4:].mean()] * len(test))
        results["Moving_Average"] = mean_absolute_error(test, sma_pred)
    except:
        results["Moving_Average"] = np.nan

    # Naive
    naive_pred = np.array([train[-1]] * len(test))
    results["Naive"] = mean_absolute_error(test, naive_pred)

    # Linear Regression (Lag Features)
    try:
        feats = create_features(series)
        X_train = feats[:len(train) - 12]
        y_train = series[12:12 + len(X_train)]
        X_test = feats[len(train) - 12:len(train) - 12 + len(test)]

        lr = LinearRegression().fit(X_train, y_train)
        lr_pred = lr.predict(X_test)
        results["Linear_Regression"] = mean_absolute_error(test, lr_pred)
    except:
        results["Linear_Regression"] = np.nan

    return results


# =====================================================================
# 5. FORECAST USING EXPONENTIAL SMOOTHING
# =====================================================================

def forecast_es(series, periods=6):
    model = ExponentialSmoothing(series, trend="add", seasonal="add", seasonal_periods=12)
    fit = model.fit()
    return fit.forecast(periods)


# =====================================================================
# 6. GRADIO UI
# =====================================================================

def create_advanced_forecasting_ui(df):

    df = preprocess(df)
    region_ts = create_demographic_ts(df, ["Region"])
    regions = sorted(region_ts["Region"].unique())

    def run_advanced_forecasting(region_name):

        series = prepare_region_ts(region_ts, region_name).values

        # Compare models
        results = compare_models(series)
        model_table = pd.DataFrame(results, index=["MAE"]).T

        # Forecast next 6 months
        future = forecast_es(series, 6)
        forecast_df = pd.DataFrame({"Month": range(1, 7), "Forecast": future})

        # Plot
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(series[-24:], label="Past 24 Months")
        ax.plot(range(len(series), len(series) + 6), future, marker="o", label="Forecast")
        ax.legend()
        ax.set_title(f"{region_name} – 6 Month Forecast")

        return (
            model_table,
            forecast_df,
            fig
        )

    ui = gr.Interface(
        fn=run_advanced_forecasting,
        inputs=gr.Dropdown(regions, label="Select Region"),
        outputs=[
            gr.Dataframe(label="Model Comparison (MAE)"),
            gr.Dataframe(label="6-Month Forecast"),
            gr.Plot(label="Forecast Plot")
        ],
        title="Advanced Forecasting — Region Based Model Comparison",
        description="Compares multiple time-series models and generates forecasts."
    )

    return ui
