from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

class LSTMTrainer:
    @staticmethod
    def create_lstm_model(input_shape):
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])
        return model

    @staticmethod
    def train_model(model, X, y, epochs=50, batch_size=32):
        history = model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2)
        return model, history

class ModelVisualizer:
    @staticmethod
    def plot_actual_vs_predicted(actual, predicted):
        plt.figure(figsize=(15, 6))
        plt.plot(actual, label='Actual')
        plt.plot(predicted, label='Predicted')
        plt.xlabel('Time Steps')
        plt.ylabel('Rainfall')
        plt.title('Actual vs Predicted Rainfall')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_loss_history(history):
        plt.figure(figsize=(15, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        plt.show()

class ModelEvaluator:
    @staticmethod
    def evaluate_model(actual, predicted):
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        r2 = r2_score(actual, predicted)
        return rmse, r2

    @staticmethod
    def print_evaluation_metrics(rmse, r2):
        print(f'Root Mean Squared Error: {rmse}')
        print(f'R-squared: {r2}')
