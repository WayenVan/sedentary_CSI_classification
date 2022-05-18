from random import shuffle
from turtle import shape

from numpy import dtype
import dataset
from  models_tf.BvP import BvP
import os
import tensorflow as tf
from models_tf.CADM import CADM
from tensorflow.data import Dataset
from dataset_tf import CatmGenerator
import json


from keras import optimizers, losses, metrics
import keras

#global parameter
# model_select = 'bvp'
model_select = 'cadm'
fraction_for_test = 0.1
# data_dir = r'dataset/BVP/6-link/user1'
data_dir = r'dataset/DAM_nonToF/all0508'
model_dir = r'./saved_models'
use_cuda = True

#training parameter
n_epochs = 10
n_batch_size = 32
learning_rate = 0.00005

#model parameter
ALL_MOTION = [0, 1, 2, 3, 4, 5, 6, 7]
N_MOTION = len(ALL_MOTION)
T_MAX = 100
img_size = (30, 30, 1)


envrionment = (1,)

"""-----------------loading data-----------------"""
data_list_origin = os.listdir(data_dir)

#select envioronment
data_list = []
for file_name in data_list_origin:
    if int(file_name.split('-')[2]) in envrionment:
        data_list.append(file_name)

#shuffle
shuffle(data_list)

data_gen = CatmGenerator(data_list, data_dir, N_MOTION, T_MAX, img_resize=(img_size[0], img_size[1]))

# def data_gen():
#     for file_name in data_list:
#         data, label = load_data_catm(data_dir, file_name, T_MAX)
#         data = tf.convert_to_tensor(data, dtype=tf.float32)
#         data = rearrange(data, 's h (w c)  -> s h w c', c=1)
#         data = tf.image.resize(data, (img_size[0], img_size[1]))
#         label = tf.convert_to_tensor(label, dtype=tf.int32)
#         yield data, label


dataset = Dataset.from_generator(
    data_gen.generator,
    output_signature=(
        tf.TensorSpec(shape=(T_MAX, img_size[0], img_size[1], img_size[2]), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
)

test_size = int(len(data_list) * fraction_for_test)
train_size = len(data_list) - test_size

train_dataset = dataset.take(train_size)
test_dataset = dataset.skip(train_size)

train_dataset = train_dataset.batch(n_batch_size)
test_dataset = test_dataset.batch(n_batch_size)


"""-----------------create model -----------------"""

if model_select == 'bvp':

    h_parameters = {
        "img_size":img_size,
        "d_model": T_MAX
    }
    with open('saved_models/'+model_select+'/h_parameters.json', 'w') as f:
        json.dump(h_parameters, f)
    
    model = BvP(T_MAX)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=[metrics.SparseCategoricalAccuracy()]
        )


if model_select == 'cadm':

    h_parameters = {
        "img_size":img_size,
        "d_model": T_MAX
    }
    with open('saved_models/'+model_select+'/h_parameters.json', 'w') as f:
        json.dump(h_parameters, f)
    
    model = CADM(T_MAX)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=[metrics.SparseCategoricalAccuracy()],
        run_eagerly=True
        )

"""--------------------training---------------------------"""
x = model(tf.random.uniform(shape=(12, T_MAX, img_size[0], img_size[1], img_size[2])))
model.summary()
print("total data amount: {}".format(len(data_list)))

checkpoint = keras.callbacks.ModelCheckpoint(
    'saved_models/'+model_select+'/checkpoint_{epoch:02d}', 
    save_weights_only=True,
    monitor = 'val_sparse_categorical_accuracy',
    mode = 'max',
    save_best_only=True
)
model.fit(x=train_dataset, epochs=100, validation_data=test_dataset, callbacks=[checkpoint])


