import numpy as np
from numpy import newaxis
from keras.layers import Dense, Dropout, LSTM, Bidirectional
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler

class Model():
    def __init__(self):
        self.model = Sequential()
        
    def load_model(self, filepath):
        self.model = load_model(filepath)
        
    def build_model(self, configs):
        for layer in configs['model']['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            input_dim = layer['input_dim'] if 'input_dim' in layer else None
            #learnin_rate = configs['model']['learnin_rate']
            metrics = [configs['model']['metrics']]
                        
            if layer['type'] == 'bidirectional': self.model.add(Bidirectional(LSTM(neurons, return_sequences=return_seq), input_shape=(input_timesteps, input_dim)))
            if layer['type'] == 'dense': self.model.add(Dense(neurons, activation=activation))
            if layer['type'] == 'lstm': self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            if layer['type'] == 'dropout': self.model.add(Dropout(dropout_rate))
            
            self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'])
             
             
    def train(self, x, y, x_test, y_test, epochs, batch_size, save_dir):  
	
        callbacks = [
                    EarlyStopping(monitor='val_loss', patience=10), 
                    ReduceLROnPlateau(monitor='loss', patience=2, cooldown=2),
                    #LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20))
                    ]
        #Add these lines if only one feature
        x = x[:, :, newaxis]
        x_test = x_test[:, :, newaxis]
        estimator = self.model.fit(x,y, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size,callbacks=callbacks, verbose=1)
		
        return estimator
    
    def predict(self, data):
        #Add this lines if only one feature
        data = data[:, :, newaxis]
        predicted = self.model.predict(data)
        predicted = np.reshape(predicted, (predicted.size,))
        return np.array(predicted)
    
    def future_prediction(self, data, window_size, future_period):
        curr_frame = data[-1]    
        predicted = []
            
        for i in range(future_period):
            predicted.append(self.model.predict(curr_frame[newaxis, :, newaxis])[0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
        return np.array(predicted)  
