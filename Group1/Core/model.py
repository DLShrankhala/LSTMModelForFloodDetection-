import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, r2_score

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
    def __init__(self, data):
        self.data = data

    def plot_actual_vs_predicted(self, actual, predicted):
        plt.figure(figsize=(15, 6))
        plt.plot(actual, label='Actual')
        plt.plot(predicted, label='Predicted')
        plt.xlabel('Time Steps')
        plt.ylabel('Rainfall')
        plt.title('Actual vs Predicted Rainfall')
        plt.legend()
        plt.show()

    def plot_loss_history(self, history):
        plt.figure(figsize=(15, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        plt.show()

def evaluate_model(model, X, y, data_normalizer, original_shape):
    predictions = model.predict(X)
    predictions = data_normalizer.inverse_transform(predictions, original_shape)
    actual = data_normalizer.inverse_transform(y.reshape(-1, 1), original_shape)

    rmse = np.sqrt(mean_squared_error(actual, predictions))
    r2 = r2_score(actual, predictions)
    print(f'Root Mean Squared Error: {rmse}')
    print(f'R-squared: {r2}')

    return actual, predictions
