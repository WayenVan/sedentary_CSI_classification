import tensorflow as tf
import keras as k
from keras import layers
from keras import models
class ImgGRU(layers.Layer):

    def __init__(self, d_model, img_size, gru_num_layers,
                 fc_hidden_size= 64, gru_hidden_size=64, dropout=0.1):

        self.gru = layers.GRU()
        self.fc = models.Sequential(
            layers.Dense(fc_hidden_size, activation='relu'),
            layers.Dense(d_model, activation='softmax')
        )


