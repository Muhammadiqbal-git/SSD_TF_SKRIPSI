import os
import tensorflow as tf
from utils import bbox_utils, data_utils, draw_utils, io_utils, train_utils, eval_utils
from models.decoder import get_decoder_model
from models.ssd_vgg16_architecture import get_model, init_model
import numpy as np


args = io_utils.handle_args()
if args.handle_gpu:
    io_utils.handle_gpu_compatibility()

batch_size = 4
evaluate = False
use_custom_images = True
use_custom_dataset = True
backbone = args.backbone
io_utils.is_valid_backbone(backbone)

hyper_params = train_utils.get_hyper_params(backbone)

par_cwd_dir = os.path.dirname(os.getcwd())
custom_data_dir = os.path.join(par_cwd_dir, 'imgs')
custom_img_dir = os.path.join(par_cwd_dir, 'custom_test_imgs')

voc_data_dir = os.path.join(par_cwd_dir, "voc_dataset")
if use_custom_dataset:
    test_data, info = data_utils.get_custom_dataset("test", custom_data_dir)
else:
    test_data, info = data_utils.get_dataset("voc/2007", "test", voc_data_dir)
total_items = data_utils.get_total_item_size(info, "test")
labels = data_utils.get_labels(info)
labels = ["bg"] + labels
print(labels)
hyper_params["total_labels"] = len(labels)
img_size = hyper_params["img_size"]

data_types = data_utils.get_data_types()
data_shapes = data_utils.get_data_shapes()
padding_values = data_utils.get_padding_values()

if use_custom_images:
    img_paths = data_utils.get_custom_imgs(custom_img_dir)
    total_items = len(img_paths)
    test_data = tf.data.Dataset.from_generator(lambda: data_utils.custom_data_generator(
                                               img_paths, img_size, img_size), data_types, data_shapes)
elif use_custom_dataset:
    print('aa')
    test_data = test_data.map(lambda x : data_utils.preprocessing(x, img_size, img_size))
else:
    test_data = test_data.map(lambda x : data_utils.preprocessing(x, img_size, img_size, evaluate=evaluate))

test_data = test_data.padded_batch(batch_size, padded_shapes=data_shapes, padding_values=padding_values)

ssd_model = get_model(hyper_params)
ssd_model_path = io_utils.get_model_path(backbone)
ssd_model.load_weights(ssd_model_path)

anchors = bbox_utils.generate_anchors(hyper_params["feature_map_shapes"], hyper_params["aspect_ratios"])
print(anchors.shape)
ssd_decoder_model = get_decoder_model(ssd_model, anchors, hyper_params)

step_size = train_utils.get_step_size(total_items, batch_size)
pred_bboxes, pred_scores, pred_labels= ssd_decoder_model.predict(test_data, steps=step_size, verbose=1)
print('----output----')
print(np.asarray(pred_bboxes))
print(np.asarray(pred_bboxes).shape)

print(np.asarray(pred_scores))
print(np.asarray(pred_scores).shape)

print(np.asarray(pred_labels))
print(np.asarray(pred_labels).shape)

print(evaluate)
if evaluate:
    eval_utils.evaluate_predictions(test_data, pred_bboxes, pred_labels, pred_scores, labels, batch_size)
else:
    draw_utils.draw_predictions(test_data, pred_bboxes, pred_labels, pred_scores, labels, batch_size)