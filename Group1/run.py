import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from Group1.Core.data import DataProcessor, FeatureEngineer, DataNormalizer, DataSplitter, DataVisualizer
from Group1.Core.model import LSTMTrainer, ModelVisualizer, ModelEvaluator
from Group1.config import Column

# Load and preprocess data
bihar_path = "path/to/dataet"
processor = DataProcessor(bihar_path)
processor.load_data()
data = processor.get_data()

# Apply feature engineering
feature_engineer = FeatureEngineer()
data = feature_engineer.create_features(data)

# Normalize data
data_normalizer = DataNormalizer()
data = data_normalizer.normalize(data)

# Filter data for a specific district
district_data = data[data[Column.DISTRICT.value].str.contains('Patna')]

# Split data
data_splitter = DataSplitter()
time_steps = 30
X, y = data_splitter.split_data(district_data[[Column.TMIN.value, Column.TMAX.value, Column.RAINFALL.value]].values, time_steps)

# Reshape data for LSTM [samples, time_steps, features]
X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))

# Create and train LSTM model
lstm_trainer = LSTMTrainer()
model = lstm_trainer.create_lstm_model((X.shape[1], X.shape[2]))

# Experiment with different batch sizes and epochs
model, history = lstm_trainer.train_model(model, X, y, epochs=200, batch_size=30)  # Adjust batch size and epochs

# Make predictions
predictions = model.predict(X)

# Inverse transform predictions
original_shape = (predictions.shape[0], 3)  # 3 columns: 'tmin', 'tmax', 'rainfall'
predictions = data_normalizer.inverse_transform(predictions, original_shape)

# Inverse transform actual data
actual = data_normalizer.inverse_transform(y.reshape(-1, 1), original_shape)

# Evaluate model
model_evaluator = ModelEvaluator()
rmse, r2 = model_evaluator.evaluate_model(actual, predictions)
model_evaluator.print_evaluation_metrics(rmse, r2)

# Plot actual vs predicted
model_visualizer = ModelVisualizer()
model_visualizer.plot_actual_vs_predicted(actual, predictions)

# Plot loss history
model_visualizer.plot_loss_history(history)
