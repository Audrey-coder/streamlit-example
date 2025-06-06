import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error

# You can also add caching if the dataset is large or training takes time
@st.cache(allow_output_mutation=True)
def load_data():
    # Replace this with your actual data loading
    # For demo, let's create dummy monthly data if dfmonthly not loaded
    # Remove below lines if you load your actual dfmonthly
    dates = pd.date_range(start='2010-01-01', periods=120, freq='M')
    exchange_rate = np.sin(np.linspace(0, 20, 120)) + 50 + np.random.normal(0, 0.5, 120)
    df = pd.DataFrame({'date': dates, 'exchange_rate': exchange_rate})
    df.set_index('date', inplace=True)
    return df

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

def build_and_train_model(X_train, y_train):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=1, epochs=10, verbose=0)
    return model

st.title("Exchange Rate Prediction with LSTM")

# Load data
dfmonthly = load_data()
st.write("### Monthly Exchange Rate Data")
st.line_chart(dfmonthly['exchange_rate'])

data = dfmonthly[['exchange_rate']].values

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

training_data_len = int(np.ceil(len(scaled_data) * .80))

train_data = scaled_data[0:training_data_len, :]
look_back = 60
test_data = scaled_data[training_data_len - look_back:, :]

X_train, y_train = create_dataset(train_data, look_back)
X_test, y_test = create_dataset(test_data, look_back)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

with st.spinner('Training LSTM model...'):
    model = build_and_train_model(X_train, y_train)

predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

mse = mean_squared_error(y_test_actual, predictions)
mae = mean_absolute_error(y_test_actual, predictions)

st.write(f"### Model Evaluation Metrics")
st.write(f"Mean Squared Error (MSE): {mse:.4f}")
st.write(f"Mean Absolute Error (MAE): {mae:.4f}")

# Plot predictions vs actual
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(y_test_actual, label='Actual')
ax.plot(predictions, label='Predicted')
ax.set_title('Exchange Rate Prediction')
ax.set_xlabel('Time Steps')
ax.set_ylabel('Exchange Rate')
ax.legend()

st.pyplot(fig)

# After you have predictions and y_test_actual computed:

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
mae = mean_absolute_error(y_test_actual, predictions)
mse = mean_squared_error(y_test_actual, predictions)

st.write("### Model Evaluation Metrics")
st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
st.write(f"Mean Squared Error (MSE): {mse:.4f}")
