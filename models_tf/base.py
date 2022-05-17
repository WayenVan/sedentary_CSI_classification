import tensorflow as tf
import tensorflow.keras
from keras import Dense

keras = tensorflow.keras

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

x = tf.random.uniform([32,64 ])
x = Dense(128, activation='relu')(x)
print(x)

