from random import shuffle

from  models_tf.BvP import BvP
import os
import tensorflow as tf
from models_tf.CADM import CADM
from tensorflow.data import Dataset
from generator import CatmGenerator
import json
from common import random_split_data_list

from keras import optimizers, losses, metrics
import keras

#global parameter
model_select = 'bvp'
# model_select = 'cadm'
fraction_for_test = 0.1
data_dir = r'dataset/DAM_nonToF/all0508'
model_dir = r'./saved_models'
use_cuda = True

#training parameter
n_epochs = 10
n_batch_size = 32
learning_rate = 0.0001

#model parameter
ALL_MOTION = [0, 1, 2, 3, 4, 5, 6, 7]
N_MOTION = len(ALL_MOTION)
T_MAX = None
img_size = (30, 30, 1)
sequence_sample_step = None


envrionment = 1

model_save_dir = os.path.join('saved_models', '{}_catm_env{}'.format(model_select, envrionment))
os.makedirs(model_save_dir+'/', exist_ok=True)

"""-----------------loading data-----------------"""
data_list_origin = os.listdir(data_dir)

#select envioronment
data_list = []
for file_name in data_list_origin:
    if int(file_name.split('-')[2]) == envrionment:
        data_list.append(file_name)

#shuffle
shuffle(data_list)

#split
train_list, test_list = random_split_data_list(data_list, fraction_for_test)

train_gen = CatmGenerator(train_list, data_dir, img_resize=(img_size[0], img_size[1]), sample_step=sequence_sample_step)
test_gen = CatmGenerator(test_list, data_dir, img_resize=(img_size[0], img_size[1]), sample_step=sequence_sample_step)

T_MAX_TRAIN = train_gen.sequence_len()
T_MAX_TEST = test_gen.sequence_len()

train_set = Dataset.from_generator(
    train_gen.generator,
    output_signature=(
        tf.TensorSpec(shape=(T_MAX_TRAIN, img_size[0], img_size[1], img_size[2]), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
).batch(n_batch_size)

test_set = Dataset.from_generator(
    test_gen.generator,
    output_signature=(
        tf.TensorSpec(shape=(T_MAX_TEST, img_size[0], img_size[1], img_size[2]), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
).batch(n_batch_size)

print('done loading data, train samples:{}, test samples:{}'.format(len(train_list), len(test_list)))

"""-----------------create model -----------------"""
if model_select == 'bvp':
    info = {
        'data_set':'CATM',
        'train_list': train_list,
        'test_list': test_list,
        'sequence_sample': sequence_sample_step,
        'model_name':model_select,
        'input_shape':[-1, img_size[0], img_size[1], img_size[2]],
        'd_model': N_MOTION
    }

    model = BvP(N_MOTION)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=[metrics.SparseCategoricalAccuracy()]
    )


if model_select == 'cadm':
    info = {
        'data_set':'CATM',
        'train_list': train_list,
        'test_list': test_list,
        'sequence_sampe': sequence_sample_step,
        'model_name':model_select,
        'input_shape':[-1, img_size[0], img_size[1], img_size[2]],
        'd_model': N_MOTION
    }

    model = CADM(N_MOTION)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=[metrics.SparseCategoricalAccuracy()],
        run_eagerly=True
        )

"""--------------------training---------------------------"""

#save model infomations
with open(os.path.join(model_save_dir, 'info.json'), 'w') as f:
        json.dump(info, f)

x = model(tf.random.uniform(shape=(12, T_MAX, img_size[0], img_size[1], img_size[2])))
model.summary()
print("total data amount: {}".format(len(data_list)))

checkpoint = keras.callbacks.ModelCheckpoint(
    model_save_dir+'/checkpoint_{epoch:02d}', 
    save_weights_only=True,
    monitor = 'val_sparse_categorical_accuracy',
    mode = 'max',
    save_best_only=True
)
model.fit(x=train_set, epochs=100, validation_data=test_set, callbacks=[checkpoint])


