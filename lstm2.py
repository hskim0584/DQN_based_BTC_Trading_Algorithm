import numpy as np
import pandas as pd
import random
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = pd.read_csv(filepath_or_buffer="BINANCE.csv", encoding="ms932", sep=",")
original_close = data['close'].values # 'close' price to compare
dates = pd.to_datetime(data['time'])
data_var = list(data)[2:] # variables for training (open ~ RSI-based MA)
dataframe = data[data_var].astype(float) # dataframe

dataframe = dataframe[:2000]
# dataframe = dataframe
print(np.shape(dataframe))
# print(dataframe.info())

scaler = StandardScaler()
scaler = scaler.fit(dataframe)
print(scaler)
scaled_data = scaler.transform(dataframe) # (2793, 8) , scaling된 dataset
print(scaled_data)
# Dividing data (train 90% , test 10%)

num_train = 2000
print(num_train)
train_data_scaled = scaled_data[0: num_train]
test_data_scaled = scaled_data[0:150]

train_dates = dates[0: num_train]
test_dates = dates[0:150]

# Data reformatting for LSTM
interval = 1 # prediction interval
history_length = 50
input_dim = 8 # [open, high, low, close, volume, volume-ma, rsi, rsi-based ma]

train_X = []
train_Y = []
test_X = []
test_Y = []

# Data convert (for LSTM) -> data decrease 100 (50 + 50)
for i in range(history_length, num_train):
    train_X.append(train_data_scaled[i - history_length : i , 0 : train_data_scaled.shape[1]]) # t ~ t+50 , 8 data(open~rsi-ma)
    train_Y.append(train_data_scaled[i : i + 1 , 3]) # t+51 , close

for i in range(history_length, len(test_data_scaled)):
    test_X.append(test_data_scaled[i - history_length : i , 0 : test_data_scaled.shape[1]])
    test_Y.append(test_data_scaled[i : i + 1 , 3])

train_X, train_Y = np.array(train_X), np.array(train_Y)
test_X, test_Y = np.array(test_X), np.array(test_Y)

# LSTM
model = Sequential()
model.add(LSTM(64, input_shape=(train_X.shape[1], train_X.shape[2]), # length = 50 , input_dim = 8
               return_sequences=True))
model.add(LSTM(32, return_sequences=False))
model.add(Dense(train_Y.shape[1]))
model.summary()

# Learning
learning_rate = 0.01
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='mse')

try:
    model.load_weights('./LSTM_WEIGHTS2.h5')
    print("Loaded model weights from disk")
except:
    print("No weights found, training model from scratch")
    # Fit the model
    history = model.fit(train_X, train_Y, epochs=500, batch_size=50, verbose=1)
    # Save model weights after training
    model.save_weights('./LSTM_WEIGHTS.h5')

    plt.plot(history.history['loss'], label='Training loss')
    # plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend()
    plt.show()


prediction = model.predict(test_X)
print(prediction.shape, test_Y.shape)


# generate array filled with means for prediction
mean_prediction = np.repeat(scaler.mean_[np.newaxis, :], prediction.shape[0], axis=0)

# substitute predictions into the first column
mean_prediction[:, 0] = np.squeeze(prediction)

# inverse transform
y_prediction = scaler.inverse_transform(mean_prediction)[:, 0]
print(y_prediction.shape)

# generate array filled with means for testY
mean_values_test_Y = np.repeat(scaler.mean_[np.newaxis, :], test_Y.shape[0], axis=0)

# substitute testY into the first column
mean_values_test_Y[:, 0] = np.squeeze(test_Y)

# inverse transform
test_Y_original = scaler.inverse_transform(mean_values_test_Y)[:, 0]
print(test_Y_original.shape)

# plotting
plt.figure()
# plot original 'Open' prices
plt.plot(dates, original_close, color='green', label='Original Close')

# plot actual vs predicted
plt.plot(test_dates[history_length:], test_Y_original, color='blue', label='Actual Close')
plt.plot(test_dates[history_length:], y_prediction, color='red', linestyle='--', label='Predicted Close')
plt.xlabel('Date')
plt.ylabel('Open Price')
plt.title('Original, Actual and Predicted Open Price')
plt.legend()
plt.show()

# Calculate the start and end indices for the zoomed plot
zoom_start = len(test_dates)- 100
zoom_end = len(test_dates)

# Create the zoomed plot
plt.figure()

# Adjust the start index for the testY_original and y_pred arrays
adjusted_start = zoom_start - history_length

plt.plot(test_dates[zoom_start:zoom_end],
         test_Y_original[adjusted_start:zoom_end - zoom_start + adjusted_start],
         color='blue',
         label='Actual Open Price')

plt.plot(test_dates[zoom_start:zoom_end],
         y_prediction[adjusted_start:zoom_end - zoom_start + adjusted_start],
         color='red',
         linestyle='--',
         label='Predicted Open Price')

plt.legend()
plt.show()
A = []
count = 0
for i in range(99):
    if (test_Y_original[i + 1] > test_Y_original[i]) & (y_prediction[i + 1] > test_Y_original[i]):
        count += 1
        A.append(test_Y_original[i])
    elif (test_Y_original[i + 1] < test_Y_original[i]) & (y_prediction[i + 1] < test_Y_original[i]):
        count += 1
        A.append(test_Y_original[i])
    else:
        A.append(None)
plt.plot(A, 'ro')
print(count)
plt.plot(y_prediction, 'r', linestyle='--', label='Predicted Close')
plt.plot(test_Y_original, 'b', label='Actual Close')
plt.legend()
plt.show()


real_result = []
predict_result = []

for i in range(test_Y.shape[0] - 1):
    if test_Y[i] < test_Y[i+1]:
        real_result.append(1)
    else:
        real_result.append(0)

    if prediction[i] < prediction[i+1]:
        predict_result.append(1)
    else:
        predict_result.append(0)

print(real_result)
print(predict_result)

total_result = 0

for i in range(len(real_result)):
    if real_result[i] == predict_result[i]:
        total_result += 1

print(total_result / len(real_result))

#def main():
    # print(scaled_data) # (2793, 8)
    # print(scaled_data[0]) # t : 0 에서의 open ~ RSI-based MA
    # print((np.shape(scaled_data))[0]) 2793
    # print(train_data_scaled.shape[1]) 8
    # print(train_X.shape, train_Y.shape) (2463, 50, 8) (2463, 1)
    # print(test_X.shape, test_Y.shape) (230, 50, 8) (230, 1)
    # print(np.shape(test_data_scaled)) (280, 8)
    # print(test_Y[229]) == print(test_data_scaled[279][3])

#if __name__ == '__main__':
#    main()