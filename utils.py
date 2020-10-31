import os

import tensorflow as tf


def get_tpu_strategy(list_tpus=True):
    tpu_address = None
    if 'COLAB_TPU_ADDR' in os.environ:
        tpu_address = 'grpc://' + os.environ['COLAB_TPU_ADDR'] # if on colab
    tpu_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_address)
    tf.config.experimental_connect_to_cluster(tpu_resolver) # connecting to the tpu cluster
    tf.tpu.experimental.initialize_tpu_system(tpu_resolver) # intializes the tpu cluster
    if list_tpus:
        print("Associated TPUs :")
        for tpu in tf.config.list_logical_devices('TPU'):
            print('\t',tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu_resolver)
    return strategy

def crop(inp_img, real_img, crop_shape):
  img_stack  = tf.stack([inp_img, real_img], axis=0)
  crop_img = tf.image.random_crop(img_stack, size=[2, *crop_shape])
  return crop_img[0], crop_img[1]
