from genericpath import exists
import os

from torch import lstm
from generator import TimeDataGenerator
from common import random_split_data_list

import tensorflow as tf
from keras.layers import LSTM, Dense, Conv2D, Flatten, MaxPool2D
from keras import optimizers, losses, metrics
import keras
import json
from tensorflow.data import Dataset

from models_tf.CADM import Residual

data_dir = "dataset/timeData"
index_name = 'dop_spec_direct'
test_split_ratio = 0.1
envrionment = 1

learning_rate = 1e-5
batch_size = 32
hidden_size = 64

if index_name == 'timeData':
    INPUT_DIM = 8
if index_name == 'dop_spec_direct':
    INPUT_DIM = 61
if index_name == 'dop_spec_ToF':
    INPUT_DIM = 61

N_MOTION = 8
T_MAX = 3000

def environment_sel(data_list, env):
    ret = []
    for file_name in data_list:
        if int(file_name.split('-')[2]) == envrionment:
            ret.append(file_name)
    return ret

"""--------load data list------------"""


data_list = os.listdir(data_dir)
data_list = environment_sel(data_list, envrionment)

train_list, test_list = random_split_data_list(data_list, test_split_ratio)
train_gen = TimeDataGenerator(train_list, data_dir, index_name)
test_gen = TimeDataGenerator(test_list, data_dir, index_name)


dataset_train = Dataset.from_generator(
    train_gen.generator,
    output_signature=(
        tf.TensorSpec(shape=(T_MAX, INPUT_DIM), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
)

dataset_test = Dataset.from_generator(
    test_gen.generator,
    output_signature=(
        tf.TensorSpec(shape=(T_MAX, INPUT_DIM), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
)

dataset_train = dataset_train.batch(batch_size)
dataset_test = dataset_test.batch(batch_size)

"""-------create model----------"""


model_save_dir = os.path.join(
    'saved_models', 
    'LSTM_{}_env{}'.format(index_name, envrionment))

info = {
    "dataset":index_name,
    "input_shape":[T_MAX, INPUT_DIM],
    "train_list":train_list,
    "test_list":test_list,
    "hidden_size":hidden_size
}

os.makedirs(os.path.dirname(model_save_dir+'/'), exist_ok=True)
with open(model_save_dir+'/h_parameters.json', 'w') as f:
    json.dump(info, f)

inputs = keras.Input(shape=(T_MAX, INPUT_DIM))
x = Conv2D(64, 3, activation='relu')(inputs)
x = Residual(64)(x)
x = MaxPool2D()(x)
x = Flatten()(x)
x = Dense(hidden_size, activation='relu')(x)
x = Dense(N_MOTION, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=x)
model.compile(
    optimizer=optimizers.Adam(learning_rate=learning_rate),
    loss=losses.SparseCategoricalCrossentropy(),
    metrics=[metrics.SparseCategoricalAccuracy()]
)
model.summary()
checkpoint = keras.callbacks.ModelCheckpoint(
    model_save_dir+'/checkpoint_{epoch:02d}', 
    save_weights_only=True,
    monitor = 'val_sparse_categorical_accuracy',
    mode = 'max',
    save_best_only=True
)
model.fit(x=dataset_train, epochs=100, validation_data=dataset_test, callbacks=[checkpoint])

