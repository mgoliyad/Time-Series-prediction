import math
import numpy as np
import pandas as pd


class DataLoader():
    
    def __init__(self, filename, split, cols):
        dataframe = pd.read_csv(filename, index_col=[0])
        dataframe = dataframe.get(cols)
        #dataframe = dataframe.iloc[::-1]
        self.min = float(dataframe.min())
        self.max = float(dataframe.max())
        dataframe = (dataframe - self.min) / (self.max - self.min)
        i_split = int(len(dataframe) * split)
        self.data_train = dataframe.values[:i_split]
        self.data_test  = dataframe.values[i_split:]
        self.len_train  = len(self.data_train)
        self.len_test   = len(self.data_test)
        self.len_train_windows = None
        
    def get_data(self, seq_len, train):
        data_x = []
        data_y = []
        
        len = self.len_train if train else self.len_test
        
        for i in range(len - seq_len):
            x, y = self._next_window(i, seq_len, train)
            data_x.append(x)
            data_y.append(y) 
        return np.array(data_x), np.array(data_y)
    
    def _next_window(self, i, seq_len,train):
        window = self.data_train[i:i+seq_len] if train else self.data_test[i:i+seq_len]
        y = np.squeeze(window)[-1]
        x = np.squeeze(window)[:-1]
        return x, y
    
    def inverse_normalize(self, y, train=False):
        inverse_normalize_y = np.array(y) * (self.max - self.min) +self.min
        return inverse_normalize_y