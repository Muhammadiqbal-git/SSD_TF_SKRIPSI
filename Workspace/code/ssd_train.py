import os
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from keras.optimizers import SGD, Adam
import augmentator
from ssd_loss import SSDLoss
from utils import bbox_utils, data_utils, io_utils, train_utils, tf_record_utils
from models.ssd_vgg16_architecture import get_model, init_model

args = io_utils.handle_args()
if args.handle_gpu:
    io_utils.handle_gpu_compatibility()

batch_size = 8
epochs = 400
load_weights = True
with_voc_2012 = False
use_custom_dataset = True
backbone = args.backbone
io_utils.is_valid_backbone(backbone)
#

hyper_params = train_utils.get_hyper_params(backbone)
#
custom_data_dir = data_utils.get_data_dir("dataset")
voc_data_dir = data_utils.get_data_dir("voc")

if use_custom_dataset:
    tf_record_utils.write_tf_record(custom_data_dir, overwrite=False)
    train_data, info = data_utils.get_custom_dataset("train", custom_data_dir, epochs)
    val_data, _ = data_utils.get_custom_dataset("validation", custom_data_dir, epochs)
    test_data, _ = data_utils.get_custom_dataset("test", custom_data_dir)
else:
    train_data, info = data_utils.get_dataset("voc/2007", "train", voc_data_dir)
    val_data, _ = data_utils.get_dataset("voc/2007", "validation", voc_data_dir)

# data_utils.preview_data(train_data)
# aa

train_total_items = data_utils.get_total_item_size(info, "train")
val_total_items = data_utils.get_total_item_size(info, "validation")
print(train_total_items, val_total_items)

if with_voc_2012 and not use_custom_dataset:
    voc_2012_data, voc_2012_info = data_utils.get_dataset("voc/2012", "train", voc_data_dir)
    voc_2012_total_items = data_utils.get_total_item_size(voc_2012_info, "train")
    train_total_items += voc_2012_total_items
    train_data = train_data.concatenate(voc_2012_data)

labels = data_utils.get_labels(info)
labels = ["bg"] + labels
print(labels)

hyper_params["total_labels"] = len(labels)
img_size = hyper_params["img_size"]

train_data = train_data.map(lambda x : data_utils.preprocessing(x, img_size, img_size, augmentator.apply))
val_data = val_data.map(lambda x : data_utils.preprocessing(x, img_size, img_size))

data_shapes = data_utils.get_data_shapes()
padding_values = data_utils.get_padding_values()
train_data = train_data.shuffle(batch_size*4).padded_batch(batch_size, padded_shapes=data_shapes, padding_values=padding_values)
val_data = val_data.padded_batch(batch_size, padded_shapes=data_shapes, padding_values=padding_values)
#
ssd_model = get_model(hyper_params)
ssd_custom_losses = SSDLoss(hyper_params["neg_pos_ratio"], hyper_params["loc_loss_alpha"])
ssd_model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss=[ssd_custom_losses.loc_loss_fn, ssd_custom_losses.conf_loss_fn])
init_model(ssd_model)

#
ssd_model_path = io_utils.get_model_path(backbone)
if load_weights:
    ssd_model.load_weights(ssd_model_path)
ssd_log_path = io_utils.get_log_path(backbone)
# We calculate anchors for one time and use it for all operations because of the all images are the same sizes
anchors = bbox_utils.generate_anchors(hyper_params["feature_map_shapes"], hyper_params["aspect_ratios"])
print("anch")
print(anchors.shape)
print("-x-")
ssd_train_feed = train_utils.generator(train_data, anchors, hyper_params)
ssd_val_feed = train_utils.generator(val_data, anchors, hyper_params)

checkpoint_callback = ModelCheckpoint(ssd_model_path, monitor="val_loss", save_best_only=True, save_weights_only=True)
tensorboard_callback = TensorBoard(log_dir=ssd_log_path)
learning_rate_callback = LearningRateScheduler(train_utils.scheduler, verbose=0)

step_size_train = train_utils.get_step_size(train_total_items, batch_size)
step_size_val = train_utils.get_step_size(val_total_items, batch_size)
ssd_model.fit(ssd_train_feed,
              steps_per_epoch=step_size_train,
              validation_data=ssd_val_feed,
              validation_steps=step_size_val,
              epochs=epochs,
              callbacks=[checkpoint_callback, tensorboard_callback, learning_rate_callback])