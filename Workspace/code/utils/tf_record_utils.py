import tensorflow as tf
import os
from PIL import Image

def _bytes_feature(val):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[val]))

def _int64_feature(val):
    return tf.train.Feature(int_64_list=tf.train.Int64List(value=[val]))

def create_tf_example(img_path, label):
    with tf.io.gfile.GFile(img_path, 'rb') as f:
        image_data = f.read()
        print(image_data)

        # feature = {
        #     'image': ,
        #     'labels': ,
        #     'objects'
        # }
def write_tf_record(dir):
    custom_img_path = os.path.join(dir, 'all_imgs')
    tf_record_fname = os.path.join(dir, 'tes.tfrecord')
    print(tf_record_fname)
    # with tf.io.TFRecordWriter()
if __name__ == '__main__':
    create_tf_example('s','l')