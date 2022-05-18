import tensorflow as tf
import keras as k
from keras.layers import Layer, Dense, GRU, TimeDistributed, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.models import Model, Sequential


class BvP(Model):

    def __init__(self, d_model,
                 fc_hidden_size=64, gru_hidden_size=64, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.cnn = TimeDistributed(
            Sequential([
                Conv2D(16, kernel_size=5, activation='relu', data_format='channels_last'),
                MaxPooling2D(pool_size=(2, 2)),
                Flatten(),
                Dense(fc_hidden_size, activation='relu'),
                Dropout(dropout),
                Dense(fc_hidden_size, activation='relu')]
            )
        )

        self.gru = GRU(gru_hidden_size, return_sequences=False)
        self.fc = Sequential([
            Dense(fc_hidden_size, activation='relu'),
            Dropout(dropout),
            Dense(d_model, activation='softmax')
        ])

    def call(self, inputs, training=None, *args, **kwargs):
        """
        :param inputs: [b, s, h, w, c]
        :param training:
        :return:
        """
        inputs = self.cnn(inputs, training=training)
        hn = self.gru(inputs, training=training)
        output = self.fc(hn, training=training)

        return output

