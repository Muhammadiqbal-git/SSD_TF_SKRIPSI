import tensorflow as tf
from keras.layers import Layer, Input, Conv2D, MaxPool2D
from keras.models import Model
from utils import bbox_utils


class SSDDecoder(Layer):

    def __init__(self, prior_boxes, variances, max_total_size=200, score_threshold=0.5, **kwargs):
        super(SSDDecoder, self).__init__(**kwargs)
        self.prior_boxes = prior_boxes
        self.variances = variances
        self.max_total_size = max_total_size
        self.score_threshold = score_threshold

    def get_config(self):
        config = super(SSDDecoder, self).get_config()
        config.update({
            "prior_boxes": self.prior_boxes.numpy(),
            "variances": self.variances,
            "max_total_size": self.max_total_size,
            "score_threshold": self.score_threshold,
        })
        return config

    def call(self, inputs, *args, **kwargs):
        pred_deltas = inputs[0]
        pred_labels = inputs[1]
        batch_size = tf.shape(pred_deltas)[0]

        pred_deltas *= self.variances
        pred_bboxes = bbox_utils.prop2abs(self.prior_boxes, pred_deltas)

        pred_labels_map = tf.expand_dims(tf.argmax(pred_labels, -1), -1)
        pred_labels = tf.where(tf.not_equal(pred_labels_map, 0), pred_labels, tf.zeros_like(pred_labels))

        pred_bboxes = tf.reshape(pred_bboxes, (batch_size, -1, 1, 4))

        final_bboxes, final_scores, final_labels = bbox_utils.nms(pred_bboxes, pred_labels, max_ouput_size_per_class=self.max_total_size, score_threshold=self.score_threshold)
        
        return final_bboxes, final_scores, final_labels
    
def get_decoder_model(base_model, prior_boxes, hyper_params):
    bboxes, scores, labels = SSDDecoder(prior_boxes, hyper_params["variances"])(base_model.output)
    return Model(inputs=base_model.input, outputs=[bboxes, scores, labels])
