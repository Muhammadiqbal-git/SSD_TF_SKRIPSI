import tensorflow as tf
import os
from PIL import Image
import numpy
import cv2
from matplotlib import pyplot as plt
import json
from collections import namedtuple
import tensorflow_datasets as tfds
import random

# print(decoded_img.shape)
# fig, ax = plt.subplots(ncols=1, figsize=(20,20))
# image_bgr = cv2.cvtColor(decoded_img.numpy(), cv2.COLOR_RGB2BGR)
# cv2.rectangle(
#     image_bgr,
#     (int(10), int(20)),
#     (int(300), int(100)),
#     (255, 0, 0),
#     2
# )
# image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
# ax.imshow((image_rgb))
# plt.show()


def _bytes_feature(val):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[val]))


def _int64_feature(val):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=val))


def _float_feature(val):
    return tf.train.Feature(float_list=tf.train.FloatList(value=val))


def convert_to_jpeg(img_path):
    """Convert given image file path to a jpeg format

    Args:
        img_path (str): file path of target image
    """
    img = Image.open(img_path)
    _index = img_path.rfind('.')
    new_path = img_path[:_index] + '.jpeg'
    img.save(new_path, format='JPEG')
    os.remove(img_path)

def get_custom_data(img_dir, resize: tuple, split_number:float = 0.7):
    """Process the data img and json from directory provided

    Args:
        img_dir (str): directory of img and json folder
        resize (tuple): scalar value that will be used as resize img
            example : (500, 600) will make the image to be 500 height pixels and 600 width pixels
        split_number (float, optional): split the dataset for train set, the rest will become test set. 
            Defaults to 0.7.

    Returns:
        data_train, data_test, categories (list, list, dictionary): processed data and dictionary of labels
    """

    height = resize[0]
    width = resize[1]
    data_list = []
    categories = {}
    data_instance = namedtuple('DataInstance', ['img_data', 'labels', 'label', 'bbox'])
    print(os.listdir(img_dir))
    for data in os.listdir(img_dir):
        if not data.endswith(".jpeg"):
            continue
        img_path = os.path.join(img_dir, data)
        json_path = os.path.join(img_dir, data.replace('.jpeg', '.json'))
        with tf.io.gfile.GFile(img_path, 'rb') as f:
            encoded_img = f.read()
            decoded_img = tf.io.decode_jpeg(encoded_img, channels=3)
            decoded_img = tf.image.convert_image_dtype(decoded_img, tf.float32)
            decoded_img = tf.image.resize(decoded_img, [height, width])
            decoded_img = tf.image.convert_image_dtype(decoded_img, tf.uint8)
            encoded_img = tf.io.encode_jpeg(decoded_img)
        with open(json_path, 'r+') as fs:
            data = json.load(fs)
            labels = {}
            label = []
            bbox = []
            max_id = 0
            for object_ in data['shapes']:
                if categories.values():
                    max_id = max(categories.values()) + 1
                if object_['label'] not in categories.keys():
                    categories[object_['label']] = max_id
                labels[object_['label']] = categories[object_['label']]
                bbx = object_['points']
                im_h = data['imageHeight']
                im_w = data['imageWidth']
                # [ymin, xmin, ymax, xmax] format
                label.append(categories[object_['label']])
                bbox.append([bbx[0][1]/im_h, bbx[0][0]/im_w, bbx[1][1]/im_h, bbx[1][0]/im_w])
        data_list.append(data_instance(img_data=encoded_img.numpy(), labels=list(labels.values()), label=label, bbox=bbox))
    print(categories)
    random.seed(17)
    random.shuffle(data_list)
    split = int(len(data_list) * split_number)
    data_train, data_test = data_list[:split], data_list[split:]
    print(len(data_train))
    print('-================-')
    print(len(data_test))
    return data_train, data_test, categories


def create_tfds_data_dict(data, categories):
    list_data = [{
        'label': data.label[idx],
        'bbox': tfds.features.BBox(ymin=data.bbox[idx][0], xmin=data.bbox[idx][1], ymax=data.bbox[idx][2], xmax=data.bbox[idx][3])
        } for idx in range(len(data.label))
    ]
    data_dict = {
        'image': data.img_data,
        'labels': data.labels,
        'objects': list_data,
    }
    print('f')
    print(list_data)
    return data_dict

def create_tfds_feature(num_classes, shard_lengths):
    # TODO ADD LABELS NAME IN LABELS KEYS
    features = tfds.features.FeaturesDict({
        'image': tfds.features.Image(
            encoding_format='jpeg',
            doc='test'
        ),
        'labels': tfds.features.Sequence(
            tfds.features.ClassLabel(
                num_classes=num_classes
            ),
        ),
        'objects': tfds.features.Sequence({
            'label': tfds.features.ClassLabel(num_classes=num_classes),
            'bbox': tfds.features.BBoxFeature()
        })
    })

    split_info = [
        tfds.core.SplitInfo(
            name= 'train', shard_lengths = [3], num_bytes = 0
        ),
        tfds.core.SplitInfo(
            name= 'test', shard_lengths = [2], num_bytes = 0
        ),
    ]
    return features, split_info

def write_tf_record(dir):
    custom_img_dir = os.path.join(dir, 'all_imgs')
    tfrecord_train_fname = os.path.join(dir, 'humanist-train.tfrecord-00000-of-00001')
    tfrecord_test_fname = os.path.join(dir, 'humanist-test.tfrecord-00000-of-00001')

    for data in os.listdir(custom_img_dir):
        if data.endswith(".json") or data.endswith(".jpeg"):
            continue
        img_path = os.path.join(custom_img_dir, data)
        convert_to_jpeg(img_path=img_path)
    data_train, data_test, categories = get_custom_data(img_dir=custom_img_dir, resize=(2, 2), split_number=0.7)
    features, split_info = create_tfds_feature(num_classes=len(categories), shard_lengths=2)
    with tf.io.TFRecordWriter(tfrecord_train_fname) as writer: # TRAIN WRITER
        for data in data_train:
            data_dict = create_tfds_data_dict(data=data, categories=categories)
            ex_byte = features.serialize_example(data_dict)
            writer.write(ex_byte)
    with tf.io.TFRecordWriter(tfrecord_test_fname) as writer: # TEST WRITER
        for data in data_test:
            data_dict = create_tfds_data_dict(data=data, categories=categories)
            ex_byte = features.serialize_example(data_dict)
            writer.write(ex_byte)

    tfds.folder_dataset.write_metadata(
        data_dir=dir,
        features=features,
        split_infos=split_info,
        filename_template=None,
    )
    # builder = tfds.builder_from_directory(dir)
    # print(builder.as_dataset())
    print('Done!')
    # with tf.io.TFRecordWriter()
