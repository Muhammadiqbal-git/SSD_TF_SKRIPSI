from models.ssd_vgg16_architecture import get_model
from models.decoder import get_decoder_model
from utils import data_utils, train_utils, io_utils, bbox_utils, draw_utils
import tensorflow as tf
import time
import os
import random


args = io_utils.handle_args()
if args.handle_gpu:
    io_utils.handle_gpu_compatibility()

custom_data_dir = data_utils.get_data_dir('dataset')
custom_img_dir = data_utils.get_data_dir('custom_test_imgs')

_, info = data_utils.get_custom_dataset('test', custom_data_dir)
labels = data_utils.get_labels(info)
labels = ["bg"] + labels
hyper_params = train_utils.get_hyper_params()
hyper_params["total_labels"] = len(labels)

ssd_model = get_model(hyper_params)
weight_path = io_utils.get_model_path(args.backbone)
ssd_model.load_weights(weight_path)

data_types = data_utils.get_data_types()
data_shapes = data_utils.get_data_shapes()
test_data = data_utils.single_custom_data_gen(custom_img_dir, 300, 300)

anchor = bbox_utils.generate_anchors(hyper_params["feature_map_shapes"], hyper_params["aspect_ratios"])
ssd_decoder_model = get_decoder_model(ssd_model, anchor, hyper_params)
data = tf.expand_dims(test_data[0], 0)

pred_bboxes, pred_scores, pred_labels = ssd_decoder_model.predict(data, verbose=1)

data = tf.squeeze(data)
pred_bboxes = tf.squeeze(pred_bboxes)
pred_scores = tf.squeeze(pred_scores)
pred_labels = tf.squeeze(pred_labels)

time_now = time.strftime("%Y-%m-%d")
model_dir = data_utils.get_data_dir('trained_model')

saved_loc = os.path.join(model_dir, "{}_Id-{}_{}.h5".format(args.backbone, random.randint(11, 99), time_now))

# draw_utils.infer_draw_predictions(data, pred_bboxes, pred_labels, pred_scores, labels)
ssd_decoder_model.save(saved_loc, save_format="h5")
if os.path.exists(saved_loc):
    print("SUCCESS! model successfully saved in {}".format(saved_loc))
