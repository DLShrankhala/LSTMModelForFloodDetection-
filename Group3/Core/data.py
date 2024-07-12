import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class Data:
    def __init__(self):
        self.dataframe = pd.DataFrame([])

    def read(self, file_name: str):
        try:
            self.dataframe = pd.read_csv(file_name)
        except FileNotFoundError:
            print(f"File {file_name} not found.")
        except pd.errors.EmptyDataError:
            print("No data found in the file.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def clean_data(self):
        self.dataframe = self.dataframe.dropna()

    def print_head(self):
        print(self.dataframe.head())
        return self.dataframe

    def print_description(self):
        print(self.dataframe.describe())

    def normalization(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        numeric_columns = self.dataframe.select_dtypes(include=['float64', 'int64']).columns
        self.dataframe[numeric_columns] = self.scaler.fit_transform(self.dataframe[numeric_columns])

    def visualize_NORMAL(self):
        if 'Dates' in self.dataframe.columns and 'NORMAL' in self.dataframe.columns and 'district' in self.dataframe.columns:
            self.dataframe['Dates'] = pd.to_datetime(self.dataframe['Dates'], format='%d-%b-%Y')

            plt.figure(figsize=(14, 7))

            for district in self.dataframe['district'].unique():
                district_data = self.dataframe[self.dataframe['district'] == district]
                plt.plot(district_data['Dates'], district_data['NORMAL'], label=f'Normal Rainfall - {district}', linestyle='-')

            plt.title('Monthly Rainfall in Districts (NORMAL)')
            plt.xlabel('Date')
            plt.ylabel('Rainfall (mm)')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.show()

        else:
            print("Column(s) not found in the dataframe")


    def visualize_ACTUAL(self):
        if 'Dates' in self.dataframe.columns and 'ACTUAL' in self.dataframe.columns and 'district' in self.dataframe.columns:
            self.dataframe['Dates'] = pd.to_datetime(self.dataframe['Dates'], format='%d-%b-%Y')

            plt.figure(figsize=(14, 7))

            for district in self.dataframe['district'].unique():
                district_data = self.dataframe[self.dataframe['district'] == district]
                plt.plot(district_data['Dates'], district_data['ACTUAL'], label=f'Actual Rainfall - {district}', linestyle='-')

            plt.title('Monthly Rainfall in Districts (ACTUAL)')
            plt.xlabel('Date')
            plt.ylabel('Rainfall (mm)')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.show()

        else:
            print("Column(s) not found in the dataframe")