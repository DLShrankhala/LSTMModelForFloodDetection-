import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class DataProcessor:
    def __init__(self, state_path):
        self.state_path = state_path
        self.data = pd.DataFrame()

    def load_data(self):
        all_data = []
        print(f"State path: {self.state_path}")

        for file in os.listdir(self.state_path):
            if file.endswith(".csv") and not file.startswith('.'):
                file_path = os.path.join(self.state_path, file)
                print(f"Loading file: {file}, full path: {file_path}")
                try:
                    df = pd.read_csv(file_path)
                    print(f"File {file} loaded successfully with shape: {df.shape}")
                    if 'DateTime' in df.columns:
                        df['DateTime'] = pd.to_datetime(df['DateTime'])
                        df.columns = ['DateTime', 'tmin', 'tmax', 'rainfall']
                        district = file.replace('_merged.csv', '')
                        df['District'] = district
                        all_data.append(df)
                    else:
                        print(f"File {file} does not match the expected format.")
                except Exception as e:
                    print(f"Failed to load file {file}: {e}")

        if all_data:
            self.data = pd.concat(all_data)
            print("Data loaded successfully.")
        else:
            print("No data loaded.")

    def get_data(self):
        if not self.data.empty:
            return self.data
        else:
            print("Data is not loaded.")
            return pd.DataFrame()

class FeatureEngineer:
    @staticmethod
    def create_features(data):
        data['Year'] = data['DateTime'].dt.year
        data['Month'] = data['DateTime'].dt.month
        data['Day'] = data['DateTime'].dt.day
        data['DayOfWeek'] = data['DateTime'].dt.dayofweek
        data['DayOfYear'] = data['DateTime'].dt.dayofyear
        data['WeekOfYear'] = data['DateTime'].dt.isocalendar().week
        return data

class DataNormalizer:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def normalize(self, data):
        data[['tmin', 'tmax', 'rainfall']] = self.scaler.fit_transform(data[['tmin', 'tmax', 'rainfall']])
        return data

    def inverse_transform(self, data, original_shape):
        temp = np.zeros(original_shape)
        temp[:, -1] = data.flatten()
        return self.scaler.inverse_transform(temp)[:, -1]

class DataSplitter:
    @staticmethod
    def split_data(data, time_steps=30):
        X, y = [], []
        for i in range(len(data) - time_steps):
            X.append(data[i:(i + time_steps), :-1])
            y.append(data[i + time_steps, -1])
        return np.array(X), np.array(y)

class DataVisualizer:
    def __init__(self, data):
        self.data = data

    def plot_rainfall(self, district):
        district_data = self.data[self.data['District'] == district]

        if district_data.empty:
            print(f"No data found for district: {district}")
            return

        plt.figure(figsize=(15, 6))
        plt.plot(district_data['DateTime'], district_data['rainfall'], label=district)
        plt.xlabel('Date')
        plt.ylabel('Rainfall')
        plt.title(f'Rainfall Over Time in {district}')
        plt.legend()
        plt.show()

    def plot_feature_distribution(self, feature):
        plt.figure(figsize=(15, 6))
        plt.hist(self.data[feature], bins=50, alpha=0.6)
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {feature}')
        plt.show()
