
# ====================================================================================================== #
# F1ScoreCallback

from keras.callbacks import Callback
from sklearn.metrics import f1_score

import numpy as np

class F1ScoreCallback(Callback):
    def __init__(self, X_val, y_val):
        self.X_val = X_val
        self.y_val = y_val
        self.f1_epoch = []
        
    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.X_val)
        y_pred = np.round(y_pred)
        y_pred = np.argmax(y_pred, axis=1)
        logs["val_f1_micro"] = f1_score(self.y_val, y_pred, average='micro')
        logs["val_f1_none"] = f1_score(self.y_val, y_pred, average=None)

# f1score callback for googlenet
class F1ScoreCallback_googlenet(Callback):
    def __init__(self, X_val, y_val):
        self.X_val = X_val
        self.y_val = y_val
        self.f1_epoch = []
        
    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.X_val)
        y_pred = y_pred[0]

        y_pred_1 = np.round(y_pred[0])
        y_pred_2 = np.round(y_pred[1])
        y_pred_3 = np.round(y_pred[2])

        y_pred_1 = np.argmax(y_pred_1, axis=1)
        y_pred_2 = np.argmax(y_pred_2, axis=1)
        y_pred_3 = np.argmax(y_pred_3, axis=1)

        logs["val_f1_1_micro"] = f1_score(self.y_val, y_pred_1, average='micro')
        logs["val_f1_1_none"] = f1_score(self.y_val, y_pred_1, average=None)
        logs["val_f1_2_micro"] = f1_score(self.y_val, y_pred_2, average='micro')
        logs["val_f1_2_none"] = f1_score(self.y_val, y_pred_2, average=None)
        logs["val_f1_3_micro"] = f1_score(self.y_val, y_pred_3, average='micro')
        logs["val_f1_3_none"] = f1_score(self.y_val, y_pred_3, average=None)

# ====================================================================================================== #
# data sampling, data split
# Done

import pandas as pd

def dataset_sampling(raw_data, sample_size=5000, random_state=None) : 
    """
    Samples the input dataset
    - uses pd.DataFrame.sample

    Return : 
    - sampled_dataset (pd.DataFrame)
    """

    if type(raw_data) != type(pd.DataFrame()) : 
        raw_data = pd.DataFrame(raw_data)
    
    return raw_data.sample(frac=sample_size/len(raw_data), random_state=random_state)

def dataset_split(raw_data) : 
    """
    Split dataset into [feature_dataset, target_dataset] and normalize.
    - feature_dataset   : shape=(num_data, 28, 28, 1)
    - target_dataset    : shape=(num_data)

    Input datasets' target data should be exist in column as named "label"

    Return : 
    - feature_dataset (np.array), target_data (np.array)
    """

    if type(raw_data) != type(pd.DataFrame()) : 
        raw_data = pd.DataFrame(raw_data)
    
    data_feature = raw_data.drop(['label'], axis=1)
    data_target = raw_data['label']

    return data_feature.values.reshape((-1,28,28,1)) / 255.0, data_target.values

# ====================================================================================================== #
# build model
# Done

from keras.layers import Input
from keras import Model

def build_model(model_class, input_shape) : 
    input_layer = Input(shape=input_shape)
    return Model(inputs=[input_layer], outputs=[model_class(input_layer)], name=model_class.__class__.__name__)

# ====================================================================================================== #