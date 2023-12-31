# NEEDS
# BBOX UTILS
# TRAIN UTILS OF HYPERPARAMS

from utils import data_utils, draw_utils
import os
from models import prediction_head, decoder


import tensorflow as tf

model_path = "D:\\1.Skripsi\\SSD_TF_SKRIPSI\\Workspace\\trained_model\\22_2023-11-22_mobilenet_v2_ID-29.h5"
custom_img_dir = data_utils.get_data_dir('inference')
img_path = os.path.join(custom_img_dir, os.listdir(custom_img_dir)[4])

data = data_utils.single_custom_data_gen(img_path, 500, 500)

labels = ['Background', 'Human']

model = tf.keras.models.load_model(model_path, compile=False)
print(type(model))

p_bbox, p_scores, p_labels = model.predict(data)


print(any(i >= 0.5 for i in tf.squeeze(p_scores)))
print(p_scores.shape)


draw_utils.infer_draw_predictions(data, p_bbox, p_labels, p_scores, labels)


