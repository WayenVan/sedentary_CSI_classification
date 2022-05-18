from common import load_data_catm

import tensorflow as tf
from einops import rearrange

class CatmGenerator():

    def __init__(self, data_list, data_dir, N_MOTION, T_MAX, img_resize=None) -> None:

        self.data_list = data_list
        self.data_dir = data_dir
        self.N_MOTION = N_MOTION
        self.T_MAX = T_MAX
        self.img_resize = img_resize


    def generator(self):
        for file_name in self.data_list:

            data, label = load_data_catm(self.data_dir, file_name, self.T_MAX)
            data = tf.convert_to_tensor(data, dtype=tf.float32)
            data = rearrange(data, 's h (w c)  -> s h w c', c=1)

            if self.img_resize != None:
                data = tf.image.resize(data, (self.img_resize[0], self.img_resize[1]))
            label = tf.convert_to_tensor(label, dtype=tf.int32)

            yield data, label
