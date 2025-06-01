import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Exchange Rate Forecasting", layout="wide")
st.title("📈 Exchange Rate Forecast with XGBoost")

# Load data from GitHub
@st.cache_data
def load_data():
    url = "dfmonthly_modelling.csv"
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    return df

df = load_data()
st.success("Data loaded successfully!")

# ===================== Model on Full Data =====================

st.header("🔍 Part 1: Model Evaluation on Historical Data")

df_model = df.drop(columns=["Date"])
X_all = df_model.drop(columns=["exchange_rate"])
y = df_model["exchange_rate"]
X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.2, random_state=42)

model_all = XGBRegressor(random_state=42)
model_all.fit(X_train, y_train)
y_pred_all = model_all.predict(X_test)

r2 = r2_score(y_test, y_pred_all)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_all))
mae = mean_absolute_error(y_test, y_pred_all)
mape = np.mean(np.abs((y_test - y_pred_all) / y_test)) * 100
mse = mean_squared_error(y_test, y_pred_all)

st.write(f"**R² Score:** {r2:.4f}")
st.write(f"**RMSE:** {rmse:.4f}")
st.write(f"**MAE:** {mae:.4f}")
st.write(f"**MAPE:** {mape:.2f}%")
st.write(f"**MSE:** {mse:.4f}")

# Plot
st.subheader("📉 Actual vs Predicted")
fig1, ax1 = plt.subplots(figsize=(12, 6))
sorted_idx = np.argsort(y_test.values)
ax1.plot(y_test.values[sorted_idx], label="Actual", color='blue')
ax1.plot(y_pred_all[sorted_idx], label="Predicted", color='orange')
ax1.set_title("Actual vs Predicted Exchange Rate")
ax1.set_xlabel("Sorted Test Samples")
ax1.set_ylabel("Exchange Rate")
ax1.legend()
ax1.grid(True)
st.pyplot(fig1)

# ===================== Forecasting =====================

st.header("🔮 Part 2: Forecast Exchange Rate for 2024–2025")

# Set index to Date
df_forecast = df.copy()
df_forecast.set_index("Date", inplace=True)

data = df_forecast[['exchange_rate']].copy()
for lag in range(1, 13):
    data[f'lag_{lag}'] = data['exchange_rate'].shift(lag)
data.dropna(inplace=True)

X_train_uni = data.drop(columns=['exchange_rate'])
y_train_uni = data['exchange_rate']

model_uni = XGBRegressor(random_state=42)
model_uni.fit(X_train_uni, y_train_uni)

# Recursive forecasting
last_known = data.iloc[-1:].copy()
future_predictions = []
future_dates = pd.date_range(start=data.index[-1] + pd.DateOffset(months=1), periods=24, freq='MS')

for date in future_dates:
    new_row = {}
    for lag in range(1, 13):
        new_row[f'lag_{lag}'] = last_known[f'lag_{lag}'].values[0] if lag > 1 else last_known['exchange_rate'].values[0]
    
    X_new = pd.DataFrame([new_row])
    y_pred = model_uni.predict(X_new)[0]
    future_predictions.append(y_pred)

    # Shift lags
    new_lags = [y_pred] + list(last_known.iloc[0, :-1].values)
    new_entry = pd.DataFrame([new_lags], columns=['exchange_rate'] + [f'lag_{i}' for i in range(1, 13)])
    last_known = new_entry.copy()

# Forecast DF
future_df = pd.DataFrame({'Date': future_dates, 'Forecasted_Exchange_Rate': future_predictions})
future_df.set_index('Date', inplace=True)

# Plot forecast
st.subheader("📆 Forecasted Exchange Rate (2024–2025)")
fig2, ax2 = plt.subplots(figsize=(14, 6))
ax2.plot(data.index, data['exchange_rate'], label='Historical', color='blue')
ax2.plot(future_df.index, future_df['Forecasted_Exchange_Rate'], label='Forecast', linestyle='--', color='green')
ax2.set_title("Exchange Rate Forecast: 2024–2025")
ax2.set_xlabel("Date")
ax2.set_ylabel("Exchange Rate")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

# Show 2024 forecast
forecast_2024 = future_df.loc['2024-01-01':'2024-12-01']
st.subheader("📋 Forecasted Rates for Jan–Dec 2024")
st.dataframe(forecast_2024.style.format("{:.4f}"))
