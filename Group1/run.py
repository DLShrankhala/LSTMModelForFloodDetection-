from Core.data import DataProcessor, FeatureEngineer, DataNormalizer, DataSplitter, DataVisualizer
from Core.model import LSTMTrainer, ModelVisualizer, evaluate_model

# Load and preprocess data
bihar_path = "path/to/Dataset"
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
district_data = data[data['District'].str.contains('Patna')]

# Split data
data_splitter = DataSplitter()
time_steps = 30
X, y = data_splitter.split_data(district_data[['tmin', 'tmax', 'rainfall']].values, time_steps)

# data visualizations
data_visualizer = DataVisualizer(data)
data_visualizer.plot_rainfall('Patna')
data_visualizer.plot_feature_distribution('rainfall')

# Reshape data for LSTM [samples, time_steps, features]
X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))

# Create and train LSTM model
lstm_trainer = LSTMTrainer()
model = lstm_trainer.create_lstm_model((X.shape[1], X.shape[2]))

# Experiment with different batch sizes and epochs
model, history = lstm_trainer.train_model(model, X, y, epochs=200, batch_size=30)  # Adjust batch size and epochs

# Evaluate model
original_shape = (X.shape[0], 3)  # 3 columns: 'tmin', 'tmax', 'rainfall'
actual, predictions = evaluate_model(model, X, y, data_normalizer, original_shape)

# Visualize results
model_visualizer = ModelVisualizer(data)
model_visualizer.plot_actual_vs_predicted(actual, predictions)
model_visualizer.plot_loss_history(history)

