import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import numpy as np
import streamlit as st
st.write("statsmodels ARIMA imported successfully!")
try:
    from statsmodels.tsa.arima.model import ARIMA
    st.success("ARIMA imported successfully!")
except Exception as e:
    st.error(f"Failed to import ARIMA: {e}")

# Load data
df = pd.read_csv("dfmonthly_modelling.csv", parse_dates=['Date'], index_col='Date')

# Split data: last 20% for testing
test_size = int(len(df) * 0.2)
train = df['exchange_rate'][:-test_size]
test = df['exchange_rate'][-test_size:]

# Fit ARIMA
model = ARIMA(train, order=(0, 1, 1))
model_fit = model.fit()
st.text(model_fit.summary())

# Forecast on test set
forecast_test = model_fit.forecast(steps=len(test))
df['forecast'] = [None] * len(train) + list(forecast_test)

# Plot original vs forecast
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['exchange_rate'], label='Original Data')
ax.plot(df['forecast'], label='Forecast', color='red', linestyle='--')
ax.axvline(x=test.index[0], color='gray', linestyle=':', label='Train/Test Split')
ax.set_title('ARIMA Forecast vs Original')
ax.legend()
st.pyplot(fig)

# Accuracy metrics
y_true = test
y_pred_manual = df['forecast'].iloc[-len(test):]

mse = mean_squared_error(y_true, y_pred_manual)
mae = mean_absolute_error(y_true, y_pred_manual)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_true, y_pred_manual) * 100
r2 = r2_score(y_true, y_pred_manual)

st.markdown("### 📊 Forecast Accuracy Metrics")
st.write(f"**MSE**: {mse:.4f}")
st.write(f"**MAE**: {mae:.4f}")
st.write(f"**RMSE**: {rmse:.4f}")
st.write(f"**MAPE**: {mape:.2f}%")
st.write(f"**R² Score**: {r2:.4f}")

# Forecast next 12 months
future_diff_forecast = model_fit.get_forecast(steps=12).predicted_mean
last_actual_value = df['exchange_rate'].iloc[-1]

future_forecast_values = [last_actual_value + future_diff_forecast.iloc[0]]
for i in range(1, len(future_diff_forecast)):
    future_forecast_values.append(future_forecast_values[i - 1] + future_diff_forecast.iloc[i])

future_index = pd.date_range(start=df.index[-1] + pd.offsets.MonthEnd(1), periods=12, freq='M')
future_forecast_series = pd.Series(future_forecast_values, index=future_index)

st.markdown("### 🔮 Forecast for Next 12 Months")
st.line_chart(future_forecast_series)
st.dataframe(future_forecast_series.rename("Forecast"))
