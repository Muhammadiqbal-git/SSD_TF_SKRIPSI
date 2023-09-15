import os
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2

def preprocessing(image_data, final_height, final_width, augmentation_fn=None, evaluate=False):
    """Image resizing operation handled before batch operations.
    inputs:
        image_data : tensorflow dataset image_data
        final_height : final image height after resizing
        final_width : final image width after resizing

    outputs:
        img : (final_height, final_width, channels)
        gt_boxes : (gt_box_size, [y1, x1, y2, x2])
        gt_labels : (gt_box_size)
    """
    img = tf.image.convert_image_dtype(image_data['image'], tf.float32)
    gt_boxes = image_data["objects"]["bbox"]
    gt_labels = tf.cast(image_data["objects"]["label"] + 1, tf.int32)

    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (final_height, final_width))
    if evaluate:
        not_diff = tf.logical_not(image_data["objects"]["is_difficult"])
        gt_boxes = gt_boxes[not_diff]
        gt_labels = gt_labels[not_diff]
    if augmentation_fn:
        img, gt_boxes = augmentation_fn(img, gt_boxes)
    return img, gt_boxes, gt_labels

def preview_data(dataset):
    n_data = 5
    fig, ax = plt.subplots(ncols=n_data, figsize=(20,20))
    for idx, data in enumerate(dataset.take(n_data)):
        print('image of ', idx+1)
        print(data['image'].shape)
        print(data['image'][300, 300, :])
        image = tf.image.convert_image_dtype(data['image'], tf.float32)
        print(image[300, 300, :])
        print(data['labels'])
        print(data['objects']['label'])
        print(data['objects']['bbox'])
        print('ss')
        bboxs = data['objects']['bbox'] # [ymin, xmax, ymax, xmax]
        height = data['image'].shape[0]
        width = data['image'].shape[1]
        image_bgr = cv2.cvtColor(image.numpy(), cv2.COLOR_RGB2BGR)
        for bbox in bboxs:
            ymin = bbox[0]*height
            xmin = bbox[1]*width
            ymax = bbox[2]*height
            xmax = bbox[3]*width

            cv2.rectangle(
                image_bgr,
                (int(xmin), int(ymin)),
                (int(xmax), int(ymax)),
                (255, 0, 0),
                2
            )
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        ax[idx].imshow((image_rgb))
        # ax.imshow((image.numpy()))
    plt.show()


def get_dataset(name, split, data_dir):
    """Get tensorflow dataset split and info.
    inputs:
        name = name of the dataset, voc/2007, voc/2012, etc.
        split = data split string, should be one of ["train", "validation", "test"]
        data_dir = read/write path for tensorflow datasets

    outputs:
        dataset = tensorflow dataset split
        info = tensorflow dataset info
    """
    assert split in ["train", "train+validation", "validation", "test"]
    dataset, info = tfds.load(name, split=split, data_dir=data_dir, with_info=True)
    print('ds info')
    print(dataset)
    print(info.features['labels'].names)
    return dataset, info

def get_custom_dataset(split, data_dir, epochs=1):
    """Get tensorflow dataset split and info.
    inputs:
        name = name of the dataset, voc/2007, voc/2012, etc.
        split = data split string, should be one of ["train", "validation", "test"]
        data_dir = read/write path for tensorflow datasets

    outputs:
        dataset = tensorflow dataset split
        info = tensorflow dataset info
    """
    assert split in ["train", "train+validation", "validation", "test"], 'split must be in format with one of these'
    dataset_builder= tfds.builder_from_directory(builder_dir=data_dir)
    dataset = dataset_builder.as_dataset(split=split)
    dataset = dataset.repeat(epochs)
    info = dataset_builder.info
    print('ds info')
    print(dataset)
    print(info.features['labels'].names)
    return dataset, info

def get_total_item_size(info, split):
    """Get total item size for given split.
    inputs:
        info = tensorflow dataset info
        split = data split string, should be one of ["train", "validation", "test"]

    outputs:
        total_item_size = number of total items
    """
    assert split in ["train", "train+validation", "validation", "test"], 'split must be in format with one of these'
    if split == "train+validation":
        return info.splits["train"].num_examples + info.splits["validation"].num_examples
    return info.splits[split].num_examples

def get_labels(info):
    """Get label names list.
    inputs:
        info = tensorflow dataset info

    outputs:
        labels = [labels list]
    """
    return info.features["labels"].names

def get_custom_imgs(custom_image_path):
    """Generating a list of images for given path.
    inputs:
        custom_image_path = folder of the custom images
    outputs:
        custom image list = [path1, path2]
    """
    img_paths = []
    for path, dir, filenames in os.walk(custom_image_path):
        for filename in filenames:
            img_paths.append(os.path.join(path, filename))
        break
    return img_paths

def custom_data_generator(img_paths, final_height, final_width):
    """Yielding custom entities as dataset.
    inputs:
        img_paths = custom image paths
        final_height = final image height after resizing
        final_width = final image width after resizing
    outputs:
        img = (final_height, final_width, depth)
        dummy_gt_boxes = (None, None)
        dummy_gt_labels = (None, )
    """
    #before
    for img_path in img_paths:
        image = Image.open(img_path)
        resized_image = tf.image.convert_image_dtype(image, tf.float32)
        resized_image = tf.image.resize_with_pad(resized_image, final_height, final_width, method=tf.image.ResizeMethod.LANCZOS3)
        yield resized_image, tf.constant([[]], dtype=tf.float32), tf.constant([], dtype=tf.int32)

def get_data_types():
    """Generating data types for tensorflow datasets.
    outputs:
        data types = output data types for (images, ground truth boxes, ground truth labels)
    """
    return (tf.float32, tf.float32, tf.int32)

def get_data_shapes():
    """Generating data shapes for tensorflow datasets.
    outputs:
        data shapes = output data shapes for (images, ground truth boxes, ground truth labels)
    """
    return ([None, None, None], [None, None], [None,])

def get_padding_values():
    """Generating padding values for missing values in batch for tensorflow datasets.
    outputs:
        padding values = padding values with dtypes for (images, ground truth boxes, ground truth labels)
    """
    return (tf.constant(0, tf.float32), tf.constant(0, tf.float32), tf.constant(-1, tf.int32))