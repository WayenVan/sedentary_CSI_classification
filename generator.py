from operator import index
from random import sample
from common import load_data_catm, load_data_timeData

import tensorflow as tf
from einops import rearrange

class CatmGenerator():

    def __init__(self, data_list, data_dir, img_resize=None, sample_step=None) -> None:

        self.data_list = data_list
        self.data_dir = data_dir
        self.img_resize = img_resize
        self.sample_step = sample_step


    def generator(self):
        for file_name in self.data_list:

            data, label = load_data_catm(self.data_dir, file_name)
            data = rearrange(data, 's h (w c)  -> s h w c', c=1)
            if self.sample_step != None:
                data = data[::self.sample_step]
            data = tf.convert_to_tensor(data, dtype=tf.float32)
            if self.img_resize != None:
                data = tf.image.resize(data, (self.img_resize[0], self.img_resize[1]))

            label = tf.convert_to_tensor(label, dtype=tf.int32)

            yield data, label
    
    def sequence_len(self):
        if self.sample_step == None:
            return 100
        else:
            return 99//self.sample_step + 1


class TimeDataGenerator():

    def __init__(self, data_list, data_dir, index_name) -> None:

        self.data_list = data_list
        self.data_dir = data_dir
        self.index_name = index_name

    def generator(self):
        for file_name in self.data_list:
            data, label = load_data_timeData(self.data_dir, file_name, self.index_name)
            data = tf.convert_to_tensor(data, dtype=tf.float32)
            label = tf.convert_to_tensor(label, dtype=tf.int32)
            yield data, label