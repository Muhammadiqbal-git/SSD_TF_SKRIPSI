import tensorflow as tf
import os
from PIL import Image

def _bytes_feature(val):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[val]))

def _int64_feature(val):
    return tf.train.Feature(int_64_list=tf.train.Int64List(val=[val]))