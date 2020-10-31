import os

import tensorflow as tf
from tensorflow.keras import Model, Sequential, layers


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
