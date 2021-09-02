import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import datetime
import time
plt.style.use('fivethirtyeight')

# Import the stock price data (From Yahoo Finance)
ticker = 'AAPL'
period1 = int(time.mktime(datetime.datetime(2014, 1, 1, 23, 59).timetuple()))
period2 = int(time.mktime(datetime.datetime(2021, 9, 1, 23, 59).timetuple()))
interval = '1d'  # 1d, 1m This checks the data daily
timestamp = 60  # time interval which affects the price of today

query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'

df = pd.read_csv(query_string)

fig1 = plt.figure(figsize=(16, 8))
plt.title('AAPL Stock Price Data')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Closing Stock Price USD($)', fontsize=18)
# plt.show()
plt.savefig('Stock Graph.png')
plt.close('all')


# Store the stock price in an array
data = df.filter(['Close'])
dataset = data.values
training_data_len = math.ceil(len(dataset)*0.8)

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))

scaled_data = scaler.fit_transform(dataset)

# Create Training Data
training_data = scaled_data[0:training_data_len, :]

# SPlit the data into x and y training data sets
x_train = []
y_train = []

for i in range(timestamp, training_data_len):
    x_train.append(training_data[i-timestamp:i, 0])
    y_train.append(training_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the x_train dataset
# Because LSTM expects 3 dimensional data with dimensions(sample size, timestep, Number of features)
# number of features in our case is 1
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
# input shape= (timesteps, number of features)
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Create test data

test_data = scaled_data[training_data_len-timestamp:, :]

x_test = []
# y_test will be the actual values
y_test = dataset[training_data_len:, :]

for i in range(timestamp, len(test_data)):
    x_test.append(test_data[i-timestamp:i, 0])

# Convert to array
x_test = np.array(x_test)

# Reshape the x_test data to fit into the LSTM model

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Obtain price predictions from the model
predictions = model.predict(x_test)
# Inverse transform the predictions (un-scaling the data)
predictions = scaler.inverse_transform(predictions)

# Evaluate the model
# Root mean square error
rmse = np.sqrt(np.mean(((predictions - y_test)**2)))
print("Root Mean Square Error= ", rmse)

# Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

# Plot the model of validations
fig2 = plt.figure(figsize=(16, 8))
plt.title('LSTM Model For Stock Price Prediction')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Closing Stock Price USD($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Training', 'Validations', 'Predictions'], loc='lower right')
# plt.show()
plt.savefig('Model Validation.png')

# Predictions for future
datalist = dataset.tolist()
no_of_days = 7
future = []

for i in range(no_of_days):
    x = datalist[-timestamp:]
    x = np.array(x)
    x = scaler.transform(x)

    X = []
    X.append(x)

    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    pred = model.predict(X)

    pred = scaler.inverse_transform(pred)

    predl = pred.tolist()
    datalist.append(predl[0])
    future.append(predl[0])

print(future)

# Store the future in a CSV file
future_data = np.array(future)
DF = pd.DataFrame(data=future_data)
DF.to_csv(
    "future_price_data.csv")


# Plot the future trend
fig3 = plt.figure(figsize=(16, 8))
plt.title('Future Predictions For Stock Price')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Future Stock Price', fontsize=18)
plt.plot(future)
# plt.show()
plt.savefig('FuturePrice.png')
