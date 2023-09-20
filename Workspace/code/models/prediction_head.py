import tensorflow as tf
from keras.layers import Layer, Input, Conv2D, MaxPool2D, Activation

@tf.keras.utils.register_keras_serializable()
class PredictionHead(Layer):
    """Concatenate all feature maps for detection

    Args:
        Layer (_type_): _description_
    """

    def __init__(self, last_dimension, **kwargs):
        super(PredictionHead, self).__init__(**kwargs)
        self.last_dimension = last_dimension

    def get_config(self):
        config = super(PredictionHead, self).get_config()
        config.update({"last_dimension": self.last_dimension})
        return config

    def call(self, inputs, *args, **kwargs):
        last_dimension = self.last_dimension
        batch_size = tf.shape(inputs[0])[0]

        outputs = []
        for conv_layer in inputs:
            outputs.append(tf.reshape(conv_layer, (batch_size, -1, last_dimension)))
        return tf.concat(outputs, axis=1)

def get_head_from_outputs(hyper_params, outputs):
    """Produce ssd bounding boxes delta and label heads.

    Args:
        hyper_params (dictionary): _description_            outputs (list): _description_

    Outputs:
        pred_deltas (concenated bbox delta head) : 
        pred_labels (concenated label head) : 

    """
    total_labels = hyper_params["total_labels"]
    len_aspect_ratio = [len(x) + 1 for x in hyper_params["aspect_ratios"]]
    labels_head = []
    bboxes_head = []

    for i, output in enumerate(outputs):
        ar = len_aspect_ratio[i]
        labels_head.append(Conv2D(ar * total_labels, (3, 3), padding="same", name="conv_labels_{}".format(i+1))(output))
        bboxes_head.append(Conv2D(ar * 4, (3, 3), padding="same", name="conv_bboxes_{}".format(i+1))(output))
    pred_labels = PredictionHead(total_labels, name="labels_head")(labels_head)
    pred_labels = Activation("softmax", name="conf")(pred_labels)
    pred_deltas = PredictionHead(4, name="loc")(bboxes_head)

    return pred_deltas, pred_labels
