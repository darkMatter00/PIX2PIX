import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model
import tensorflow_addons as tfa

def add_layer(filters, kernel_size,batchnorm = True):
  init = tf.keras.initializers.random_normal(0., 0.02)
  gamma_init = tf.keras.initializers.random_normal(0., 0.02)
  blocks = Sequential()
  blocks.add(layers.Conv2D(filters, kernel_size=kernel_size, strides = 2, kernel_initializer=init,padding='same', use_bias=False))
  if batchnorm:
    blocks.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))
  blocks.add(layers.LeakyReLU())
  return blocks
