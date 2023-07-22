import tensorflow as tf
from keras.layers import Layer, Input, Conv2D, MaxPool2D, Activation
from keras.models import Model
from keras.regularizers import L2
from keras.applications import VGG16
from models.prediction_head import get_head_from_outputs
# header


def get_model(hyper_params):
    """Generate ssd model and hyper params

    Args:
        hyper_params (dictionary): dictionary of parameter

    Output:
        ssd_model (tf.keras.Model): a ssd model with backbone vgg16
    """

    scale_factor = 20.0
    reg_factor = 5e-4
    total_labels = hyper_params["total_labels"]
    # +1 for ratio 1 based in the original ssd paper
    len_aspect_ratio = [len(x) + 1 for x in hyper_params["aspect_ratios"]]
    # vgg_model = VGG16(input_shape=(None, None, 3), include_top=False)
    # conv_4_3 = vgg_model.get_layer("block4_conv3").output
    # conv_5_3 = vgg_model.get_layer("block5_conv3").output
    input = Input(shape=(None, None, 3), name='input')
    # conv1 block
    conv1_1 = Conv2D(64, (3, 3), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=L2(reg_factor), name="conv1_1")(input)
    conv1_2 = Conv2D(64, (3, 3), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=L2(reg_factor), name="conv1_2")(conv1_1)
    pool1 = MaxPool2D((2, 2), strides=(2, 2), padding="same", name="pool1")(conv1_2)
    # conv2 block
    conv2_1 = Conv2D(128, (3, 3), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=L2(reg_factor), name="conv2_1")(pool1)
    conv2_2 = Conv2D(128, (3, 3), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=L2(reg_factor), name="conv2_2")(conv2_1)
    pooL2 = MaxPool2D((2, 2), strides=(2, 2), padding="same", name="pooL2")(conv2_2)
    # conv3 block
    conv3_1 = Conv2D(256, (3, 3), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=L2(reg_factor), name="conv3_1")(pooL2)
    conv3_2 = Conv2D(256, (3, 3), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=L2(reg_factor), name="conv3_2")(conv3_1)
    conv3_3 = Conv2D(256, (3, 3), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=L2(reg_factor), name="conv3_3")(conv3_2)
    pool3 = MaxPool2D((2, 2), strides=(2, 2), padding="same", name="pool3")(conv3_3)
    # conv4 block
    conv4_1 = Conv2D(512, (3, 3), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=L2(reg_factor), name="conv4_1")(pool3)
    conv4_2 = Conv2D(512, (3, 3), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=L2(reg_factor), name="conv4_2")(conv4_1)
    conv4_3 = Conv2D(512, (3, 3), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=L2(reg_factor), name="conv4_3")(conv4_2)
    pool4 = MaxPool2D((2, 2), strides=(2, 2), padding="same", name="pool4")(conv4_3)
    # conv5 block
    conv5_1 = Conv2D(512, (3, 3), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=L2(reg_factor), name="conv5_1")(pool4)
    conv5_2 = Conv2D(512, (3, 3), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=L2(reg_factor), name="conv5_2")(conv5_1)
    conv5_3 = Conv2D(512, (3, 3), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=L2(reg_factor), name="conv5_3")(conv5_2)
    pool5 = MaxPool2D((3, 3), strides=(1, 1), padding="same", name="pool5")(conv5_3)
    # conv6 and conv7 converted from fc6 and fc7 and remove dropouts
    conv6 = Conv2D(1024, (3, 3), dilation_rate=6, padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=L2(reg_factor), name="conv6")(pool5)
    conv7 = Conv2D(1024, (1, 1), strides=(1, 1), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=L2(reg_factor), name="conv7")(conv6)
    ############################ Extra Feature Layers Start ############################
    # conv8 block <=> conv6 block in paper caffe implementation
    conv8_1 = Conv2D(256, (1, 1), strides=(1, 1), padding="valid", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=L2(reg_factor), name="conv8_1")(conv7)
    conv8_2 = Conv2D(512, (3, 3), strides=(2, 2), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=L2(reg_factor), name="conv8_2")(conv8_1)
    # conv9 block <=> conv7 block in paper caffe implementation
    conv9_1 = Conv2D(128, (1, 1), strides=(1, 1), padding="valid", activation="relu",
                     kernel_initializer="glorot_normal", kernel_regularizer=L2(reg_factor), name="conv9_1")(conv8_2)
    conv9_2 = Conv2D(256, (3, 3), strides=(2, 2), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=L2(reg_factor), name="conv9_2")(conv9_1)
    # conv10 block <=> conv8 block in paper caffe implementation
    conv10_1 = Conv2D(128, (1, 1), strides=(1, 1), padding="valid", activation="relu",
                      kernel_initializer="glorot_normal", kernel_regularizer=L2(reg_factor), name="conv10_1")(conv9_2)
    conv10_2 = Conv2D(256, (3, 3), strides=(1, 1), padding="valid", activation="relu",
                      kernel_initializer="glorot_normal", kernel_regularizer=L2(reg_factor), name="conv10_2")(conv10_1)
    # conv11 block <=> conv9 block in paper caffe implementation
    conv11_1 = Conv2D(128, (1, 1), strides=(1, 1), padding="valid", activation="relu",
                      kernel_initializer="glorot_normal", kernel_regularizer=L2(reg_factor), name="conv11_1")(conv10_2)
    conv11_2 = Conv2D(256, (3, 3), strides=(1, 1), padding="valid", activation="relu",
                      kernel_initializer="glorot_normal", kernel_regularizer=L2(reg_factor), name="conv11_2")(conv11_1)
    ############################ Extra Feature Layers End ############################
    # L2 normalization for each location in the feature map
    #
    pred_deltas, pred_labels = get_head_from_outputs(hyper_params, [conv9_2, conv10_2, conv11_2])
    return Model(inputs=input, outputs=[pred_deltas, pred_labels])

def init_model(model):
    """Initiate model with dummy data for load weight with optimizer state and graph construction

    Args:
        model (tf.keras.Model): _description_
    """
    model(tf.random.uniform((1, 300, 300, 3)))
