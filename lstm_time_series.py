import os
import json
import numpy as np
import matplotlib.pyplot as plt
from data_reader import DataLoader
from model import Model

def plot(predicted_data, true_data, label1, label2):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label=label1)
    plt.plot(predicted_data, label=label2)
    plt.legend()
    plt.show()

def main():
    configs = json.load(open('config.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])
    
    data = DataLoader(
        os.path.join('data', configs['data']['filename']),
        configs['data']['train_test_split'],
        configs['data']['columns']
    )
    
    model = Model()
    
    model.build_model(configs)
    
    x_test, y_test = data.get_data(seq_len=configs['data']['sequence_length'], train=False)
    
    x, y = data.get_data(seq_len=configs['data']['sequence_length'], train=True)
    
    estimator = model.train(
	x,
	y,
        x_test, 
        y_test,
	epochs = configs['training']['epochs'],
	batch_size = configs['training']['batch_size'],
	save_dir = configs['model']['save_dir'])
    
    #plot(estimator.history['acc'], estimator.history['val_acc'], label1='Train acc', label2='Test acc')
    plot(estimator.history['loss'], estimator.history['val_loss'], label1='Train loss', label2='Test loss')
   
    predictions = model.predict(x_test)
    
    predictions = data.inverse_normalize(predictions)    
    y_test = data.inverse_normalize(np.squeeze(y_test))

    furure_prediction = model.future_prediction(x_test, configs['data']['sequence_length'], configs['data']['future_period'])
    furure_prediction = data.inverse_normalize(furure_prediction) 
    furure_prediction = np.concatenate((predictions, furure_prediction), axis=None)

    y_test = np.squeeze(y_test)

    plot(furure_prediction, y_test, label1='Prediction', label2='True values')
    
if __name__ == '__main__':
    main()
