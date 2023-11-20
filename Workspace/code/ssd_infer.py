import os
import tensorflow as tf
from utils import bbox_utils, data_utils, draw_utils, io_utils, train_utils, eval_utils
from models.decoder import get_decoder_model
import numpy as np
import random


args = io_utils.handle_args()
if args.handle_gpu:
    io_utils.handle_gpu_compatibility()
backbone = args.backbone
io_utils.is_valid_backbone(backbone)

if backbone == "mobilenet_v2":
    from models.ssd_mobilenet_v2 import get_model, init_model
else:
    from models.ssd_vgg16_architecture import get_model, init_model

BATCH_SIZE = 4
EVAL_mAP = True
PENGUJIAN = True
use_custom_dataset = True
use_custom_images = True
hyper_params = train_utils.get_hyper_params(backbone)

custom_data_dir = data_utils.get_data_dir("custom_dataset")
custom_img_dir = data_utils.get_data_dir("inference")
pengujian_dir = []
for list_name in os.listdir(custom_img_dir):
    if list_name.endswith(".jpeg"):
        continue
    pengujian_dir.append(os.path.join(custom_img_dir, list_name))
#test data dari pembuatan tf_record dataset (random dari distribusi yang sama)
test_data, info, total_items = data_utils.get_custom_dataset("test", custom_data_dir)
labels = data_utils.get_labels(info)
labels = ["background"] + labels
hyper_params["total_labels"] = len(labels)
img_size = hyper_params["img_size"]

data_types = data_utils.get_data_types()
data_shapes = data_utils.get_data_shapes()
padding_values = data_utils.get_padding_values()

ssd_model = get_model(hyper_params)
ssd_model_path = io_utils.get_model_path(backbone)
ssd_model.load_weights(ssd_model_path)

anchors = bbox_utils.generate_anchors(hyper_params["feature_map_shapes"], hyper_params["aspect_ratios"])
ssd_decoder_model = get_decoder_model(ssd_model, anchors, hyper_params)


if use_custom_images:
    for pengujian in pengujian_dir :
        img_paths = data_utils.get_custom_imgs(pengujian, pengujian=PENGUJIAN)
        total_items = len(img_paths)
        step_size = train_utils.get_step_size(total_items, BATCH_SIZE)
        print(pengujian)
        print("total data uji {}".format(total_items))
        test_data = tf.data.Dataset.from_generator(lambda: data_utils.custom_data_generator(
                                                img_paths, img_size[1], img_size[0], labels, PENGUJIAN), data_types, data_shapes)
        test_data = test_data.padded_batch(BATCH_SIZE, padded_shapes=data_shapes, padding_values=padding_values)
        # print("info")
        # for idx, data in enumerate(test_data):
        #     data_utils.preview_data_inf(data)
        #     img_data, bbox, labelsz = data
        #     print(img_data.shape)
        #     print(bbox.shape)
        #     print(bbox)
        #     print(labelsz.shape)
        #     print(labelsz)
        # print("pred info")
        # a
        pred_bboxes, pred_scores, pred_labels = ssd_decoder_model.predict(test_data, steps=step_size, verbose=1)
        if EVAL_mAP:
            eval_utils.evaluate_predictions(test_data, pred_bboxes, pred_labels, pred_scores, labels, BATCH_SIZE)
        else:
            draw_utils.draw_predictions(test_data, pred_bboxes, pred_labels, pred_scores, labels, BATCH_SIZE)
else:
    print("total data {}".format(total_items))
    test_data = test_data.map(lambda x : data_utils.preprocessing(x, img_size[1], img_size[0]))
    test_data = test_data.padded_batch(BATCH_SIZE, padded_shapes=data_shapes, padding_values=padding_values)
    step_size = train_utils.get_step_size(total_items, BATCH_SIZE)
    pred_bboxes, pred_scores, pred_labels = ssd_decoder_model.predict(test_data, steps=step_size, verbose=1)
    if EVAL_mAP:
        eval_utils.evaluate_predictions(test_data, pred_bboxes, pred_labels, pred_scores, labels, BATCH_SIZE)
    else:
        draw_utils.draw_predictions(test_data, pred_bboxes, pred_labels, pred_scores, labels, BATCH_SIZE)

