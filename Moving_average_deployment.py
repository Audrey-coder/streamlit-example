# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.title("📈 Exchange Rate Forecasting using SARIMAX")

# --- Load data from repo ---
@st.cache_data
def load_data():
    df = pd.read_csv("dfmonthly_modelling.csv", parse_dates=['Date'], index_col='Date')
    return df

dfmonthly = load_data()

# Select exogenous and target variables
target = 'exchange_rate'
exog_vars = ['Exports ', 'Imports ', 'Total Debt', 'Total Remittances ',
             'IBRD loans and IDA credits (DOD, current US$)',
             'Unemployment Rate', 'Deposit']

y = dfmonthly[target]
X = dfmonthly[exog_vars]

# Train-test split
train_size = len(dfmonthly) - 12
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]

# Stationarity check
adf_result = adfuller(y_train.dropna())
p_value = adf_result[1]
d = 1 if p_value > 0.05 else 0
st.write(f"ADF p-value: {p_value:.4f} ⇒ Differencing required: {d}")

# Grid search for (p,d,q)
best_score, best_cfg = float("inf"), None
for p in range(0, 3):
    for q in range(0, 3):
        try:
            model = SARIMAX(y_train, exog=X_train, order=(p, d, q),
                            enforce_stationarity=False, enforce_invertibility=False)
            results = model.fit(disp=False)
            forecast = results.get_forecast(steps=12, exog=X_test)
            forecast_mean = forecast.predicted_mean
            rmse = np.sqrt(mean_squared_error(y_test, forecast_mean))
            if rmse < best_score:
                best_score, best_cfg = rmse, (p, d, q)
        except:
            continue

st.write(f"✅ Best SARIMAX Order: {best_cfg} with RMSE={best_score:.4f}")
best_p, best_d, best_q = best_cfg

# Train final model
model = SARIMAX(y_train, exog=X_train, order=(best_p, best_d, best_q),
                enforce_stationarity=False, enforce_invertibility=False)
results = model.fit()
forecast = results.get_forecast(steps=12, exog=X_test)
forecast_mean = forecast.predicted_mean

# Evaluation
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

mae = mean_absolute_error(y_test, forecast_mean)
mse = mean_squared_error(y_test, forecast_mean)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, forecast_mean)
mape = mean_absolute_percentage_error(y_test, forecast_mean)

st.subheader("📊 Evaluation Metrics")
st.write(f"**MAE:** {mae:.4f}")
st.write(f"**MSE:** {mse:.4f}")
st.write(f"**RMSE:** {rmse:.4f}")
st.write(f"**R²:** {r2:.4f}")
st.write(f"**MAPE:** {mape:.2f}%")

# Plot actual vs forecasted
st.subheader("📉 Actual vs Forecasted Exchange Rate")
fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(y_train, label='Training Data')
ax1.plot(y_test, label='Actual Test Data')
ax1.plot(forecast_mean, label='Forecasted Data', color='red')
ax1.set_title('Actual vs Forecasted Exchange Rate')
ax1.legend()
ax1.grid(True)
st.pyplot(fig1)

# Refit on all data
future_exog = X.iloc[-12:]
final_model = SARIMAX(y, exog=X, order=(best_p, best_d, best_q),
                      enforce_stationarity=False, enforce_invertibility=False).fit()
future_forecast = final_model.get_forecast(steps=12, exog=future_exog)
future_index = pd.date_range(start=y.index[-1] + pd.offsets.MonthEnd(1), periods=12, freq='M')
future_series = pd.Series(future_forecast.predicted_mean.values, index=future_index)

# Forecast plot
st.subheader("📈 12-Month Future Forecast")
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(y, label='Historical Data')
ax2.plot(future_series, label='12-Month Forecast', color='green')
ax2.set_title('12-Month Exchange Rate Forecast')
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

# Forecast table
st.subheader("📅 Forecasted Values")
st.write(future_series)

