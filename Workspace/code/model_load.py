# NEEDS
# BBOX UTILS
# TRAIN UTILS OF HYPERPARAMS

from utils import train_utils, bbox_utils, data_utils, draw_utils
from models import prediction_head, decoder


import tensorflow as tf

model_path = "D:\\1.Skripsi\\SSD_VGG_TF_SKRIPSI\\Workspace\\trained_model\\vgg16_Id-57_2023-09-20.h5"
custom_img_dir = data_utils.get_data_dir('custom_test_imgs')

test_data = data_utils.single_custom_data_gen(custom_img_dir, 300, 300)
data = tf.expand_dims(test_data[0], 0)

labels = ['background', 'face']

model = tf.keras.models.load_model(model_path)

p_bbox, p_scores, p_labels = model.predict(data)
data = tf.squeeze(data)
p_bbox = tf.squeeze(p_bbox)
p_scores = tf.squeeze(p_scores)
p_labels = tf.squeeze(p_labels)

draw_utils.infer_draw_predictions(data, p_bbox, p_labels, p_scores, labels)


