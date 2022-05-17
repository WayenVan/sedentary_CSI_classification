from turtle import shape

from numpy import dtype
import dataset
from  models_tf.BvP import BvP
import os
import tensorflow as tf
from tensorflow.data import Dataset
from common import load_data_BvP
from dataset import BvPDataset
from einops import rearrange

from keras import optimizers, losses, metrics

# Parameters
# model_select = 'bvp'
model_select = 'img_gru'
fraction_for_test = 0.1
data_dir = r'dataset/BVP/6-link/user1'
# data_dir = r'dataset/DAM_nonToF/all0508'
model_dir = r'./saved_models'
ALL_MOTION = [0,1,2,3,4, 5]
N_MOTION = len(ALL_MOTION)
T_MAX = 30
n_epochs = 10
f_dropout_ratio = 0.5
n_batch_size = 32
n_test_batch_size = 64
f_learning_rate = 0.001
img_size = (20, 20, 1)
envrionment = (1,)
use_cuda = True
log_interval = 10
dry_run = False

#-----------------loading data-----------------#
    


data_list_origin = os.listdir(data_dir)

#select envioronment
data_list = data_list_origin
# for file_name in data_list_origin:
#     if int(file_name.split('-')[2]) in envrionment:
#         data_list.append(file_name)
def data_gen():
    for file_name in data_list:
        data, label = load_data_BvP(data_dir, file_name, T_MAX)
        data = tf.convert_to_tensor(data, dtype=tf.float64)
        data = rearrange(data, '(c h) w s -> s h w c', c=1)
        label = tf.convert_to_tensor(label, dtype=tf.int32)
        yield data, label



dataset = Dataset.from_generator(
    data_gen,
    output_signature=(
        tf.TensorSpec(shape=(T_MAX, img_size[0], img_size[1], img_size[2]), dtype=tf.float64),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
)


# Split train and test



"""-----------------create model -----------------"""

model = BvP(T_MAX)
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.5),
    loss=losses.SparseCategoricalCrossentropy(),
    metrics=[metrics.SparseCategoricalAccuracy()]
    )

x = model(tf.random.uniform(shape=(32, T_MAX, 20, 20, 1)))
model.summary()
model.fit(x=dataset.batch(n_batch_size), epochs=100)