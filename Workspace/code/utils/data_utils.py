import os
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2
from json import load as json_load


def preprocessing(data, final_height, final_width, augmentation_fn=None, evaluate=False):
    """Image resizing operation handled before batch operations.

    this function also convert the text label to an +1 integer e.g ['bike', 'car', 'person'] to [1, 2, 3]
    Args:
        image_data : tensorflow dataset 
        final_height : final image height after resizing
        final_width : final image width after resizing

    Returns:
        img : (final_height, final_width, channels)
        gt_boxes : (gt_box_size, [y1, x1, y2, x2])
        gt_labels : (gt_box_size)
    """
    img = tf.image.convert_image_dtype(data["image"], tf.float32)
    gt_boxes = data["objects"]["bbox"]
    gt_labels = tf.cast(data["objects"]["label"] + 1, tf.int32)
    print(img)
    img = tf.image.resize(img, (final_height, final_width))
    if evaluate:
        not_diff = tf.logical_not(data["objects"]["is_difficult"])
        gt_boxes = gt_boxes[not_diff]
        gt_labels = gt_labels[not_diff]
    if augmentation_fn:
        img, gt_boxes = augmentation_fn(img, gt_boxes)
    return img, gt_boxes, gt_labels


def preview_data(dataset):
    n_data = 4
    fig, ax = plt.subplots(ncols=n_data, figsize=(20, 20))
    for idx, data in enumerate(dataset.take(n_data)):
        print("image of ", idx+1)
        print(data["image"].shape)
        print(data["image"][300, 300, :])
        image = tf.image.convert_image_dtype(data["image"], tf.float32)
        print(image[300, 300, :])
        print("labels = {}".format(data["labels"]))
        print("label = {}".format(data["objects"]["label"]))
        print("bbox")
        print(data["objects"]["bbox"])
        print("ss")
        bboxs = data["objects"]["bbox"]  # [ymin, xmax, ymax, xmax]
        height = data["image"].shape[0]
        width = data["image"].shape[1]
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


def preview_data_inf(dataset):
    fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
    img, bboxs, labels = dataset
    for batch_idx in range(img.shape[0]):
        # print("image of ", batch_idx+1)
        # print(img[batch_idx, ...].shape)
        image = tf.image.convert_image_dtype(img[batch_idx, :, :], tf.float32)
        # print("labels = {}".format(labels))
        # print("bboxs")
        # print(bboxs)
        # print("ss")
        height = img[batch_idx, ...].shape[0]
        width = img[batch_idx, ...].shape[1]
        image_bgr = cv2.cvtColor(image.numpy(), cv2.COLOR_RGB2BGR)
        for bbox in bboxs[batch_idx, ...]:
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
        ax[batch_idx].imshow((image_rgb))
        # ax.imshow((image.numpy()))
    plt.show()


def get_data_dir(subset):
    """get dataset directory

    Args:
        subset (String): get dataset directory in subfolder "code", should be one of ["custom_dataset", "inference", "voc"]
                        if not provided, it will get the "subset" folder within the same level.

    Returns:
        String: a directory of dataset.
    """
    par_dir = os.path.dirname(os.getcwd())
    if subset == "custom_dataset":
        return os.path.join(par_dir, "imgs")
    elif subset == "inference":
        return os.path.join(par_dir, "inference_test_imgs")
    elif subset == "voc":
        return os.path.join(par_dir, "voc_dataset")
    else:
        return os.path.join(par_dir, subset)


def get_dataset(name, split, data_dir):
    """Get tensorflow dataset split and info.
    Args:
        name = name of the dataset, voc/2007, voc/2012, etc.
        split = data split string, should be one of ["train", "validation", "test"]
        data_dir = read/write path for tensorflow datasets

    Returns:
        dataset = tensorflow dataset split
        info = tensorflow dataset info
    """
    assert split in ["train", "train+validation", "validation", "test"]
    dataset, info = tfds.load(name, split=split, data_dir=data_dir, with_info=True)
    return dataset, info


def get_custom_dataset(split, data_dir, epochs=1):
    """Get tensorflow dataset split and info.
    Args:
        name = name of the dataset, voc/2007, voc/2012, etc.
        split = data split string, should be one of ["train", "validation", "test"]
        data_dir = read/write path for tensorflow datasets

    Returns:
        dataset = tensorflow dataset split
        info = tensorflow dataset info
    """
    assert split in ["train", "train+validation", "validation", "test"], "split must be in format with one of these"
    dataset_builder = tfds.builder_from_directory(builder_dir=data_dir)
    dataset = dataset_builder.as_dataset(split=split)
    dataset = dataset.repeat(1)
    info = dataset_builder.info
    print("ds info")
    print(dataset)
    print(info.features["labels"].names)
    return dataset, info, info.splits[split].num_examples


def get_total_item_size(info, split):
    """Get total item size for given split.
    Args:
        info = tensorflow dataset info
        split = data split string, should be one of ["train", "validation", "test"]

    Returns:
        total_item_size = number of total items
    """
    assert split in ["train", "train+validation", "validation", "test"], "split must be in format with one of these"
    if split == "train+validation":
        return info.splits["train"].num_examples + info.splits["validation"].num_examples
    return info.splits[split].num_examples


def get_labels(info):
    """Get label names list.
    Args:
        info = tensorflow dataset info

    Returns:
        labels = [labels list]
    """
    return info.features["labels"].names


def get_custom_imgs(custom_image_path, pengujian):
    """Generating a list of images for given path.
    Args:
        custom_image_path = folder of the custom images
    Returns:
        custom image list = [path1, path2]
    """
    img_paths = []
    if pengujian:
        for filename in os.listdir(custom_image_path):
            if filename.endswith(".json"):
                continue
            img_paths.append(os.path.join(custom_image_path, filename))
    else:
        for path, dir, filenames in os.walk(custom_image_path):
            for filename in filenames:

                img_paths.append(os.path.join(path, filename))
            break
    return img_paths


def single_custom_data_gen(img_data, final_height, final_width):
    """Yielding single custom entities as dataset.
    Args:
        img_paths = custom image paths
        final_height = final image height after resizing
        final_width = final image width after resizing
    Returns:
        img = (final_height, final_width, depth)
        dummy_gt_boxes = (None, None)
        dummy_gt_labels = (None, )
    """
    image = Image.open(img_data)
    resized_image = tf.image.convert_image_dtype(image, tf.float32)
    resized_image = tf.image.resize_with_pad(resized_image, final_height, final_width, method=tf.image.ResizeMethod.LANCZOS3)
    data = tf.expand_dims(resized_image, 0)
    return data


def custom_data_generator(img_paths, final_height, final_width, labels, with_label: bool = False):
    """Yielding custom entities as dataset.
    Args:
        img_paths = custom image paths
        final_height = final image height after resizing
        final_width = final image width after resizing
    Returns:
        img = (final_height, final_width, depth)
        dummy_gt_boxes = (None, None)
        dummy_gt_labels = (None, )
    """
    dict_label = {}
    for idx, label in enumerate(labels):
        dict_label[label] = idx 
        print("aaa")
        print(dict_label)
    for img_path in img_paths:
        json_path = img_path.replace(".jpeg", ".json")
        image = Image.open(img_path)
        resized_image = tf.image.convert_image_dtype(image, tf.float32)
        resized_image = tf.image.resize_with_pad(resized_image, final_height, final_width, method=tf.image.ResizeMethod.LANCZOS3)
        if with_label:
            with open(json_path, "r+") as fs:
                data = json_load(fs)
                label = []
                bbox = []
                for object_ in data["shapes"]:
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
                    bbox.append(                    [
                            (bbx[0][1] + pad_size_y) / im_h,
                            (bbx[0][0] + pad_size_x) / im_w,
                            (bbx[1][1] + pad_size_y) / im_h,
                            (bbx[1][0] + pad_size_x) / im_w,
                        ])
                    label.append(dict_label[object_["label"]])
            yield resized_image, bbox, label
        else:
            yield resized_image, tf.constant([[]], dtype=tf.float32), tf.constant([], dtype=tf.int32)


def get_data_types():
    """Generating data types for tensorflow datasets.
    Returns:
        data types = output data types for (images, ground truth boxes, ground truth labels)
    """
    return (tf.float32, tf.float32, tf.int32)


def get_data_shapes():
    """Generating data shapes for tensorflow datasets.
    Returns:
        data shapes = output data shapes for (images, ground truth boxes, ground truth labels)
    """
    return ([None, None, None], [None, None], [None, ])


def get_padding_values():
    """Generating padding values for missing values in batch for tensorflow datasets.
    Returns:
        padding values = padding values with dtypes for (images, ground truth boxes, ground truth labels)
    """
    return (tf.constant(0, tf.float32), tf.constant(0, tf.float32), tf.constant(-1, tf.int32))
