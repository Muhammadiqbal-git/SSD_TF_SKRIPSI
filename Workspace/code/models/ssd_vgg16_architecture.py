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
    reg_factor = 5e-4
    img_size = hyper_params["img_size"]
    # +1 for ratio 1 based in the original ssd paper
    vgg = VGG16(input_shape=(img_size[1], img_size[0], 3), include_top=False)
    input_ = vgg.input
    conv4_3 = vgg.get_layer("block4_conv3").output
    # pool4 = MaxPool2D((2, 2), strides=(2, 2), padding="same", name="pool4")(conv4_3)
    # input_ = Input(shape=(None, None, 3), name="input")
    # # conv1 block - Halved
    # conv1_1 = Conv2D(64, (3, 3), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=L2(reg_factor), name="conv1_1")(input_)
    # conv1_2 = Conv2D(64, (3, 3), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=L2(reg_factor), name="conv1_2")(conv1_1)
    # pool1 = MaxPool2D((2, 2), strides=(2, 2), padding="same", name="pool1")(conv1_2)
    # # conv2 block
    # conv2_1 = Conv2D(128, (3, 3), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=L2(reg_factor), name="conv2_1")(pool1)
    # conv2_2 = Conv2D(128, (3, 3), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=L2(reg_factor), name="conv2_2")(conv2_1)
    # pool2 = MaxPool2D((2, 2), strides=(2, 2), padding="same", name="pool2")(conv2_2)
    # # conv3 block - Halved
    # conv3_1 = Conv2D(128, (3, 3), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=L2(reg_factor), name="conv3_1")(pool2)
    # conv3_2 = Conv2D(256, (3, 3), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=L2(reg_factor), name="conv3_2")(conv3_1)
    # conv3_3 = Conv2D(256, (3, 3), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=L2(reg_factor), name="conv3_3")(conv3_2)
    # pool3 = MaxPool2D((2, 2), strides=(2, 2), padding="same", name="pool3")(conv3_3)
    # # conv4 block 
    # conv4_1 = Conv2D(512, (3, 3), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=L2(reg_factor), name="conv4_1")(pool3)
    # conv4_2 = Conv2D(512, (3, 3), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=L2(reg_factor), name="conv4_2")(conv4_1)
    # conv4_3 = Conv2D(512, (3, 3), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=L2(reg_factor), name="conv4_3")(conv4_2)
    pool4 = MaxPool2D((2, 2), strides=(2, 2), padding="same", name="pool4")(conv4_3)
    # conv5 block - Halved start here
    conv5_1 = Conv2D(256, (3, 3), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=L2(reg_factor), name="conv5_1")(pool4)
    conv5_2 = Conv2D(256, (3, 3), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=L2(reg_factor), name="conv5_2")(conv5_1)
    conv5_3 = Conv2D(256, (3, 3), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=L2(reg_factor), name="conv5_3")(conv5_2)
    pool5 = MaxPool2D((3, 3), strides=(1, 1), padding="same", name="pool5")(conv5_3)
    # conv6 and conv7 converted from fc6 and fc7 and remove dropouts
    conv6 = Conv2D(512, (3, 3), dilation_rate=6, padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=L2(reg_factor), name="conv6")(pool5)
    conv7 = Conv2D(512, (1, 1), strides=(1, 1), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=L2(reg_factor), name="conv7")(conv6)
    ############################ Extra Feature Layers Start ############################
    # conv8 block <=> conv6 block in paper caffe implementation
    conv8_1 = Conv2D(128, (1, 1), strides=(1, 1), padding="valid", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=L2(reg_factor), name="conv8_1")(conv7)
    conv8_2 = Conv2D(256, (3, 3), strides=(2, 2), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=L2(reg_factor), name="conv8_2")(conv8_1)
    # conv9 block <=> conv7 block in paper caffe implementation
    conv9_1 = Conv2D(64, (1, 1), strides=(1, 1), padding="valid", activation="relu",
                     kernel_initializer="glorot_normal", kernel_regularizer=L2(reg_factor), name="conv9_1")(conv8_2)
    conv9_2 = Conv2D(128, (3, 3), strides=(2, 2), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=L2(reg_factor), name="conv9_2")(conv9_1)
    # conv10 block <=> conv8 block in paper caffe implementation
    conv10_1 = Conv2D(64, (1, 1), strides=(1, 1), padding="valid", activation="relu",
                      kernel_initializer="glorot_normal", kernel_regularizer=L2(reg_factor), name="conv10_1")(conv9_2)
    conv10_2 = Conv2D(128, (3, 3), strides=(1, 1), padding="valid", activation="relu",
                      kernel_initializer="glorot_normal", kernel_regularizer=L2(reg_factor), name="conv10_2")(conv10_1)
    # conv11 block <=> conv9 block in paper caffe implementation
    conv11_1 = Conv2D(64, (1, 1), strides=(1, 1), padding="valid", activation="relu",
                      kernel_initializer="glorot_normal", kernel_regularizer=L2(reg_factor), name="conv11_1")(conv10_2)
    conv11_2 = Conv2D(128, (3, 3), strides=(1, 1), padding="valid", activation="relu",
                      kernel_initializer="glorot_normal", kernel_regularizer=L2(reg_factor), name="conv11_2")(conv11_1)
    ############################ Extra Feature Layers End ############################
    # ALL UNIT ARE HALVED BECAUSE LOW SPEC PC
    # L2 normalization for each location in the feature map
    #

    pred_deltas, pred_labels = get_head_from_outputs(hyper_params, [conv4_3, conv7, conv8_2, conv9_2, conv10_2, conv11_2])
    # return Model(inputs=input_, outputs=[conv4_3, conv7, conv8_2, conv9_2, conv10_2, conv11_2])
    return Model(inputs=input_, outputs=[pred_deltas, pred_labels])


def init_model(model):
    """Initiate model with dummy data for load weight with optimizer state and graph construction

    Args:
        model (tf.keras.Model): _description_
    """
    tes = model(tf.random.uniform((1, 500, 500, 3)))
    # print(tes[0].shape)
    # print(tes[1].shape)
    # print(tes[2].shape)
    # print(tes[3].shape)
    # print(tes[4].shape)
    # print(tes[5].shape)
    # a
