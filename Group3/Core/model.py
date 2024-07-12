from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Training:
    def __init__(self, dataframe, scaler):
        self.dataframe = dataframe
        self.scaler = scaler

    def prepare_lstm_data(self, seq_length=3):
        self.dataframe['Dates'] = pd.to_datetime(self.dataframe['Dates'])
        self.dataframe['Year'] = self.dataframe['Dates'].dt.year
        self.dataframe['Month'] = self.dataframe['Dates'].dt.month

        features = self.dataframe[['Year', 'Month', 'NORMAL', 'ACTUAL']]
        features_scaled = self.scaler.fit_transform(features)

        def create_sequences(data, seq_length):
            xs, ys = [], []
            for i in range(len(data) - seq_length):
                x = data[i:i + seq_length]
                y = data[i + seq_length][3]
                xs.append(x)
                ys.append(y)
            return np.array(xs), np.array(ys)

        self.X, self.y = create_sequences(features_scaled, seq_length)
        print("Shape of X:", self.X.shape)
        print("Shape of y:", self.y.shape)
        

    def build_lstm_model(self):
        self.model = Sequential()
        self.model.add(LSTM(50, return_sequences=True, input_shape=(self.X.shape[1], self.X.shape[2])))
        self.model.add(LSTM(50, return_sequences=False))
        self.model.add(Dense(25))
        self.model.add(Dense(1))

        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.summary()


    def train_lstm_model(self, epochs=50, batch_size=1):
        train_size = int(len(self.X) * 0.8)
        self.X_train, self.X_test = self.X[:train_size], self.X[train_size:]
        self.y_train, self.y_test = self.y[:train_size], self.y[train_size:]

        self.history = self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, validation_data=(self.X_test, self.y_test))

        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper right')
        plt.show()


    def evaluate_lstm_model(self):
        y_pred = self.model.predict(self.X_test)

        y_test_scaled = self.scaler.inverse_transform(np.concatenate((np.zeros((self.y_test.shape[0], 3)), self.y_test.reshape(-1, 1)), axis=1))[:, 3]
        y_pred_scaled = self.scaler.inverse_transform(np.concatenate((np.zeros((y_pred.shape[0], 3)), y_pred), axis=1))[:, 3]

        rmse = np.sqrt(np.mean((y_test_scaled - y_pred_scaled) ** 2))
        print('Root Mean Squared Error:', rmse)

        plt.plot(y_test_scaled, label='True Value')
        plt.plot(y_pred_scaled, label='Predicted Value')
        plt.title('Predictions vs True Values')
        plt.xlabel('Time Step')
        plt.ylabel('ACTUAL')
        plt.legend()
        plt.show()