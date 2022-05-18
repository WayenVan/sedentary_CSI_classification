import tensorflow as tf
import keras as k
from keras.layers import Layer, Dense, GRU, TimeDistributed, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Add, ReLU, LSTM
from keras.models import Model, Sequential
from torch import dropout, relu

class CADM(Model):

    def __init__(self, d_model, filters=64, fc_hidden_size=64, lstm_hidden_size=64, dropout=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.res_cnn = TimeDistributed(
            Sequential([
                Conv2D(filters, 3, activation='relu'),
                Residual(filters),
                Residual(filters),
                MaxPooling2D((2, 2)),
                Flatten(),
                Dropout(dropout),
        ]), name='cnn')

        self.lstm = LSTM(lstm_hidden_size, return_sequences=False, return_state=False)

        self.fc = Sequential([
            Dense(fc_hidden_size, activation='relu'),
            Dropout(dropout),
            Dense(d_model, activation='softmax')
        ], name='fc')

    def call(self, inputs, training=None, mask=None):
        """
        :param inputs [b, s, h, w, c]
        """
        
        x = self.res_cnn(inputs, training=training)
        x = self.lstm(x, training=training)
        x = self.fc(x,training=training )

        return x


class Residual(Layer):
    
    def __init__(self, filters, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.block = Sequential([
            BatchNormalization(),
            ReLU(),
            Conv2D(filters, 3, padding='same'),
            BatchNormalization(),
            ReLU(),
            Conv2D(filters, 3, padding='same')
        ])

        self.add = Add()

    
    def call(self, inputs, training=None, mask=None):
        """
        :param inputs [b, h, w, c]
        """

        x = self.block(inputs)
        x = self.add([inputs, x])
        return x 



        
