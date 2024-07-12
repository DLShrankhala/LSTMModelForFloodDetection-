# run.py

from Core.data import Data
from Core.model import Training
from sklearn.preprocessing import MinMaxScaler

def main():
    data = Data()
    
    file_name = 'Data/Odisha Rainfall day-wise.csv'
    data.read(file_name)
    data.clean_data()
    
    data.print_head()
    data.print_description()
    
    data.normalization()
    
    data.visualize_NORMAL()
    data.visualize_ACTUAL()
    
    training = Training(data.dataframe, data.scaler)
    
    training.prepare_lstm_data(seq_length=3)
    
    training.build_lstm_model()
    training.train_lstm_model(epochs=100, batch_size=64)
    
    training.evaluate_lstm_model()

if __name__ == '__main__':
    main()
