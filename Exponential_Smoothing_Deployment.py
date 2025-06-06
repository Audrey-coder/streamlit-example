import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.title("Exchange Rate Forecast with Exponential Smoothing")

# Load data directly from GitHub or another repo
url = "dfmonthly_modelling.csv"

try:
    model_df = pd.read_csv(url)
    model_df['Date'] = pd.to_datetime(model_df['Date'])
    model_df.sort_values('Date', inplace=True)
    model_df.set_index('Date', inplace=True)

    # Fit Exponential Smoothing Model
    es_model = ExponentialSmoothing(
        model_df['exchange_rate'],
        trend='add',
        seasonal=None,
        initialization_method='estimated'
    ).fit()

    model_df['exchange_rate_es'] = es_model.fittedvalues

    # Performance metrics
    mse = mean_squared_error(model_df['exchange_rate'], model_df['exchange_rate_es'])
    mae = mean_absolute_error(model_df['exchange_rate'], model_df['exchange_rate_es'])
    r2 = r2_score(model_df['exchange_rate'], model_df['exchange_rate_es'])

    st.subheader("Model Performance on Historical Data")
    st.write(f"MSE: {mse:.4f}")
    st.write(f"MAE: {mae:.4f}")
    st.write(f"R2: {r2:.4f}")

    # Forecast next 12 months
    forecast_steps = 12
    forecast_values = es_model.forecast(steps=forecast_steps)
    forecast_index = pd.date_range(start=model_df.index[-1] + pd.offsets.MonthEnd(1), periods=forecast_steps, freq='M')
    forecast_df = pd.DataFrame({'exchange_rate_forecast': forecast_values.values}, index=forecast_index)

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(model_df.index, model_df['exchange_rate'], label='Actual', color='purple')
    plt.plot(model_df.index, model_df['exchange_rate_es'], label='Fitted (ES)', color='orange')
    plt.plot(forecast_df.index, forecast_df['exchange_rate_forecast'], label='Forecast', color='red', linestyle='--')
    plt.title('Exchange Rate: Actual, Fitted, and 12-Month Forecast (Exponential Smoothing)')
    plt.xlabel('Date')
    plt.ylabel('Exchange Rate')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)

    st.subheader("12-Month Forecast")
    st.dataframe(forecast_df)

except Exception as e:
    st.error(f"Failed to load or process the data: {e}")
