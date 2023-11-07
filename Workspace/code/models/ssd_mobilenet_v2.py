import tensorflow as tf
from keras.layers import Layer, Input, Conv2D, MaxPool2D, Activation
from keras.models import Model
from keras.applications import MobileNetV2
from models.prediction_head import get_head_from_outputs
# header
# THIS ARCHITECTURE TOO BIG FOR GTX1050 3GB AND I5 7500 3.4 GHZ

def get_model(hyper_params):
    """Generate ssd model and hyper params

    Args:
        hyper_params (dictionary): dictionary of parameter

    Output:
        ssd_model (tf.keras.Model): a ssd model with backbone vgg16
    """
    img_size = hyper_params["img_size"]
    # +1 for ratio 1 based in the original ssd paper
    base_model = MobileNetV2(input_shape=(img_size[1], img_size[0], 3), include_top=False)
    input_ = base_model.input
    conv_fm1 = base_model.get_layer("block_13_expand_relu").output
    conv_fm2 = base_model.output
    # extra conv layer
    conv_fm3_1 = Conv2D(256, (1, 1), strides=(1, 1), padding="valid", activation="relu", name="conv_fm3_1")(conv_fm2)
    conv_fm3_2 = Conv2D(512, (3, 3), strides=(2, 2), padding="same", activation="relu", name="conv_fm3_2")(conv_fm3_1)
    #
    conv_fm4_1 = Conv2D(128, (1, 1), strides=(1, 1), padding="valid", activation="relu", name="conv_fm4_1")(conv_fm3_2)
    conv_fm4_2 = Conv2D(256, (3, 3), strides=(2, 2), padding="same", activation="relu", name="conv_fm4_2")(conv_fm4_1)
    #
    conv_fm5_1 = Conv2D(128, (1, 1), strides=(1, 1), padding="valid", activation="relu", name="conv_fm5_1")(conv_fm4_2)
    conv_fm5_2 = Conv2D(256, (3, 3), strides=(2, 2), padding="same", activation="relu", name="conv_fm5_2")(conv_fm5_1)
    #
    conv_fm6_1 = Conv2D(128, (1, 1), strides=(1, 1), padding="valid", activation="relu", name="conv_fm6_1")(conv_fm5_2)
    conv_fm6_2 = Conv2D(256, (3, 3), strides=(2, 2), padding="same", activation="relu", name="conv_fm6_2")(conv_fm6_1)
    

    pred_deltas, pred_labels = get_head_from_outputs(hyper_params, [conv_fm1, conv_fm2, conv_fm3_2, conv_fm4_2, conv_fm5_2, conv_fm6_2])
    # return Model(inputs=input_, outputs=[conv_fm1, conv_fm2, conv_fm3_2, conv_fm4_2, conv_fm5_2, conv_fm6_2])
    return Model(inputs=input_, outputs=[pred_deltas, pred_labels])


def init_model(model):
    """Initiate model with dummy data for load weight with optimizer state and graph construction

    Args:
        model (tf.keras.Model): _description_
    """
    _model = model(tf.random.uniform((1, 500, 500, 3)))
    # print(_model[0].shape)
    # print(_model[1].shape)
    # print(_model[2].shape)
    # print(_model[3].shape)
    # print(_model[4].shape)
    # print(_model[5].shape)
    # a
