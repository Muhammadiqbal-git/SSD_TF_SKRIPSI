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

def convert_to_jpeg(img_path):
    """Convert given image file path to a jpeg format

    Args:
        img_path (str): file path of target image
    """
    img = Image.open(img_path)
    _index = img_path.rfind(".")
    new_path = img_path[:_index] + ".jpeg"
    img.save(new_path, format="JPEG")
    os.remove(img_path)


def resize_img(img_path, resize: tuple):
    width = resize[0]
    height = resize[1]
    with tf.io.gfile.GFile(img_path, "rb") as f:
        encoded_img = f.read()
        decoded_img = tf.io.decode_jpeg(encoded_img, channels=3)
        decoded_img = tf.image.convert_image_dtype(decoded_img, tf.float32)
        decoded_img = tf.image.resize_with_pad(decoded_img, height, width, method=tf.image.ResizeMethod.LANCZOS3)
        decoded_img = tf.image.convert_image_dtype(decoded_img, tf.uint8)
        encoded_img = tf.io.encode_jpeg(decoded_img)
    return encoded_img



def get_custom_data(img_dir, resize: tuple, split_number: tuple = (0.6, 0.2)):
    """Process the data img and json from directory provided

    (If there is no bounding box defined, it will create a default bbox consist of [0, 0, 0, 0] and with 'bg' as the label)

    Args:
        img_dir (str): directory of img and json folder
        resize (tuple): scalar value that will be used as resize img
            example : (500, 600) will make the image to be 500 height pixels and 600 width pixels
        split_number (tuple, optional): split the dataset for train and validation set,
            the rest will become test set.
            Defaults to (0.7, 0.2).

    Returns:
        data_train, data_test, categories (list, list, dictionary): processed data and dictionary of labels
    """

    assert (
        split_number[0] + split_number[1] <= 100
    ), "Split sum value must be below than 1.0"
    print(split_number[0] + split_number[1])
    data_list = []
    categories = {}
    data_instance = namedtuple("DataInstance", ["img_data", "labels", "label", "bbox"])
    # print(os.listdir(img_dir))
    for data in os.listdir(img_dir):
        if not data.endswith(".jpeg"):
            continue
        img_path = os.path.join(img_dir, data)
        json_path = os.path.join(img_dir, data.replace(".jpeg", ".json"))
        encoded_img = resize_img(img_path=img_path, resize=resize)
        with open(json_path, "r+") as fs:
            data = json.load(fs)
            labels = {}
            label = []
            bbox = []
            max_id = 0
            if data["shapes"] == []:
                print(data)
                raise Exception("Data kosong terdeteksi")
            for object_ in data["shapes"]:
                if categories.values():
                    max_id = max(categories.values()) + 1
                if object_["label"] not in categories.keys():
                    categories[object_["label"]] = max_id
                labels[object_["label"]] = categories[object_["label"]]
                bbx = object_["points"]
                im_w = data["imageWidth"]
                im_h = data["imageHeight"] 
                pad_size_x = 0
                pad_size_y = 0
                if im_w > im_h:
                    pad_size_y = (im_w - im_h) / 2
                    im_h += im_w - im_h
                else:
                    pad_size_x = (im_h - im_w) / 2
                    im_w += im_h - im_w
                # [ymin, xmin, ymax, xmax] format
                label.append(categories[object_["label"]])
                bbox.append(
                    [
                        (bbx[0][1] + pad_size_y) / im_h,
                        (bbx[0][0] + pad_size_x) / im_w,
                        (bbx[1][1] + pad_size_y) / im_h,
                        (bbx[1][0] + pad_size_x) / im_w,
                    ]
                )
                if (bbx[0][1] + pad_size_y) / im_h >= (bbx[1][1] + pad_size_y) / im_h:
                    print(data)
                    print("aaa")
                    break
                if (bbx[0][0] + pad_size_x) / im_w >= (bbx[1][0] + pad_size_x) / im_w:
                    print(data)
                    print("aaas")
                    break
        data_list.append(
            data_instance(
                img_data=encoded_img.numpy(),
                labels=list(labels.values()),
                label=label,
                bbox=bbox,
            )
        )
        # print(data_list)
        # print(categories)
        # aa
    random.seed(42)
    random.shuffle(data_list)
    categories = dict(sorted(categories.items(), key=lambda x:x[1]))
    split_train = int(len(data_list) * split_number[0] / 100)
    split_val = int(len(data_list) * split_number[1] / 100) + split_train


    # aa
    data_train, data_val, data_test = (
        data_list[:split_train],
        data_list[split_train:split_val],
        data_list[split_val:],
    )
    return data_train, data_val, data_test, categories


def create_tfds_data_dict(data, categories):
    list_data = [
        {
            "label": data.label[idx],
            "bbox": tfds.features.BBox(
                ymin=data.bbox[idx][0],
                xmin=data.bbox[idx][1],
                ymax=data.bbox[idx][2],
                xmax=data.bbox[idx][3],
            ),
        }
        for idx in range(len(data.label))
    ]
    data_dict = {
        "image": data.img_data,
        "labels": data.labels,
        "objects": list_data,
    }

    return data_dict


def create_tfds_feature(name_classes, num_classes, shard_lengths: tuple):
    features = tfds.features.FeaturesDict(
        {
            "image": tfds.features.Image(encoding_format="jpeg", doc="test"),
            "labels": tfds.features.Sequence(
                tfds.features.ClassLabel(
                    names=name_classes,
                    ),
            ),
            "objects": tfds.features.Sequence(
                {
                    "label": tfds.features.ClassLabel(
                        names=name_classes,
                        ),
                    "bbox": tfds.features.BBoxFeature(),
                }
            ),
        }
    )

    split_info = [
        tfds.core.SplitInfo(
            name="train", shard_lengths=[shard_lengths[0]], num_bytes=0
        ),
        tfds.core.SplitInfo(name="validation", shard_lengths=[shard_lengths[1]], num_bytes=0),
        tfds.core.SplitInfo(name="test", shard_lengths=[shard_lengths[2]], num_bytes=0),
    ]
    return features, split_info


def write_tf_record(dir, overwrite, img_size: tuple):
    """Write dataset in TF Record format

    Args:
        dir (str): target directory of written dataset, must have 'src_imgs' folder as the images will be used.
        overwrite (bool): wheter or not the tfrecord dataset will be overwritten.
    """
    custom_img_dir = os.path.join(dir, "src_imgs")
    tfrecord_train_fname = os.path.join(dir, "customist-train.tfrecord-00000-of-00001")
    tfrecord_val_fname = os.path.join(dir, "customist-validation.tfrecord-00000-of-00001")
    tfrecord_test_fname = os.path.join(dir, "customist-test.tfrecord-00000-of-00001")
    if (
        os.path.exists(tfrecord_train_fname)
        and os.path.exists(tfrecord_test_fname)
        and not overwrite
    ):
        return

    for data in os.listdir(custom_img_dir):
        if data.endswith(".json") or data.endswith(".jpeg"):
            continue
        img_path = os.path.join(custom_img_dir, data)
        convert_to_jpeg(img_path=img_path)

    data_train, data_val, data_test, categories = get_custom_data(
        img_dir=custom_img_dir, resize=img_size, split_number=(70, 20)
    )
    print(categories)
    features, split_info = create_tfds_feature(
        name_classes=list(categories.keys()),
        num_classes=len(categories),
        shard_lengths=(len(data_train), len(data_val), len(data_test)),
    )
    # write the .tfrecord file
    for split_of_data, split_of_fname in zip(
        (data_train, data_val, data_test),
        (tfrecord_train_fname, tfrecord_val_fname, tfrecord_test_fname),
    ):
        with tf.io.TFRecordWriter(split_of_fname) as writer:
            for data in split_of_data:
                data_dict = create_tfds_data_dict(data=data, categories=categories)
                ex_byte = features.serialize_example(data_dict)
                writer.write(ex_byte)
    # write the tfrecord metadata
    tfds.folder_dataset.write_metadata(
        data_dir=dir,
        features=features,
        split_infos=split_info,
        filename_template=None,
    )
    # builder = tfds.builder_from_directory(dir)
    # print(builder.as_dataset())
    print("Generating custom TF Record Done!")
    # with tf.io.TFRecordWriter()