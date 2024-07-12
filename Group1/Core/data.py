import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from Group1.config import Column

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
                    if Column.DATE.value in df.columns:
                        df[Column.DATE.value] = pd.to_datetime(df[Column.DATE.value])
                        df.columns = [Column.DATE.value, Column.TMIN.value, Column.TMAX.value, Column.RAINFALL.value]
                        district = file.replace('_merged.csv', '')
                        df[Column.DISTRICT.value] = district
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
        data['Year'] = data[Column.DATE.value].dt.year
        data['Month'] = data[Column.DATE.value].dt.month
        data['Day'] = data[Column.DATE.value].dt.day
        data['DayOfWeek'] = data[Column.DATE.value].dt.dayofweek
        data['DayOfYear'] = data[Column.DATE.value].dt.dayofyear
        data['WeekOfYear'] = data[Column.DATE.value].dt.isocalendar().week
        return data

class DataNormalizer:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def normalize(self, data):
        data[[Column.TMIN.value, Column.TMAX.value, Column.RAINFALL.value]] = self.scaler.fit_transform(
            data[[Column.TMIN.value, Column.TMAX.value, Column.RAINFALL.value]])
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
        district_data = self.data[self.data[Column.DISTRICT.value] == district]

        if district_data.empty:
            print(f"No data found for district: {district}")
            return

        plt.figure(figsize=(15, 6))
        plt.plot(district_data[Column.DATE.value], district_data[Column.RAINFALL.value], label=district)
        plt.xlabel('Date')
        plt.ylabel('Rainfall')
        plt.title(f'Rainfall Over Time in {district}')
        plt.legend()
        plt.show()
