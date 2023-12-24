import tensorflow as tf
import math
from utils import bbox_utils

SSD = {
    "vgg16": {
        "img_size": [500, 500],
        "feature_map_shapes": [62, 31, 16, 8, 6, 4],
        "aspect_ratios": [[1., 2./3., 1./2.],
                         [1., 2., 1./2., 2./3., 1./3.],
                         [1., 2., 1./2., 2./3., 1./3.],
                         [1., 2., 1./2., 2./3., 1./3.],
                         [1., 2., 1./2.],
                         [1., 2., 1./2.]],
    },
    "mobilenet_v2": {
        "img_size": [500, 500],
        "feature_map_shapes": [32, 16, 8, 4, 2, 1],
        "aspect_ratios": [[1., 1./3., 1./2.],
                         [1., 1./4., 1./2., 2./3., 1./3.],
                         [1., 1./4., 1./2., 2./3., 1./3.],
                         [1., 1./4., 1./2., 2./3., 1./3.],
                         [1., 2./3., 1./2.],
                         [1., 2./3., 1./2.]],
    },
}
loaded_weight = False

def get_hyper_params(backbone="mobilenet_v2", **kwargs):
    """Generating hyper params in a dynamic way.
    inputs:
        **kwargs : any value could be updated in the hyper_params, default to 'vgg16'

    outputs:
        hyper_params : dictionary
    """
    hyper_params = SSD[backbone]
    hyper_params["iou_threshold"] = 0.50
    hyper_params["neg_pos_ratio"] = 2
    hyper_params["loc_loss_alpha"] = 1
    hyper_params["variances"] = [0.1, 0.1, 0.2, 0.2]
    for key, value in kwargs.items():
        if key in hyper_params and value:
            hyper_params[key] = value
    #
    return hyper_params

def scheduler(epoch):
    """Generating learning rate value for a given epoch.
    inputs:
        epoch : number of current epoch

    outputs:
        learning_rate : float learning rate value
    """
    if loaded_weight:
        return 1e-5
    if epoch < 25:
        return 1e-3
    elif epoch < 50:
        return 1e-4
    elif epoch < 100:
        return 1.5e-4
    elif epoch < 200:
        return 1e-5
    else:
        return 1e-6

def get_step_size(total_items, batch_size):
    """Get step size for given total item size and batch size.
    inputs:
        total_items : number of total items
        batch_size : number of batch size during training or validation

    outputs:
        step_size : number of step size for model training
    """
    return math.ceil(total_items / batch_size)

def generator(dataset, anchors, hyper_params):
    """Tensorflow data generator for fit method, yielding inputs and outputs.
    inputs:
        dataset : tf.data.Dataset, PaddedBatchDataset
        anchors : (total_anchors, [ymin, xmin, ymax, xmax])
            these values in normalized format between [0, 1]
        hyper_params : dictionary

    outputs:
        yield inputs, outputs
    """
    while True:
        for image_data in dataset:
            img, gt_boxes, gt_labels = image_data
            # print(f"gen gt labels {gt_labels.shape}")
            # print(f"gen gt labels {gt_labels[0, :]}")
            # print(f"gen gt box {gt_boxes.shape}")
            actual_deltas, actual_labels = calculate_actual_outputs(anchors, gt_boxes, gt_labels, hyper_params)
            # print(f"img deltas {actual_deltas}")
            # print(f"img deltas {actual_deltas.shape}")

            yield img, (actual_deltas, actual_labels)

def calculate_actual_outputs(anchors, gt_boxes, gt_labels, hyper_params):
    """Calculate actual outputs that same as SSD Model outputs format.
    Batch operations supported.
    inputs:
        anchors : (total_anchors, [ymin, xmin, ymax, xmax])
            these values in normalized format between [0, 1]
        gt_boxes (batch_size, gt_box_size, [ymin, xmin, ymax, xmax])
            these values in normalized format between [0, 1]
        gt_labels (batch_size, gt_box_size)
        hyper_params : dictionary

    outputs:
        bbox_deltas : (batch_size, total_bboxes, [delta_y, delta_x, delta_h, delta_w])
        bbox_labels : (batch_size, total_bboxes, [0,0,...,0])
    """
    debug = True
    batch_size = tf.shape(gt_boxes)[0]
    total_labels = hyper_params["total_labels"]
    iou_threshold = hyper_params["iou_threshold"]
    variances = hyper_params["variances"]
    # total_anchors = anchors.shape[0]
    # Calculate iou values between each bboxes and ground truth boxes
    iou_map = bbox_utils.compute_iou(anchors, gt_boxes)
    # Get max index value for each row
    max_indices_each_gt_box = tf.argmax(iou_map, axis=2, output_type=tf.int32)
    # IoU map has iou values for every gt boxes and we merge these values column wise
    merged_iou_map = tf.reduce_max(iou_map, axis=2)
    #
    pos_cond = tf.greater(merged_iou_map, iou_threshold)
    #
    gt_boxes_map = tf.gather(gt_boxes, max_indices_each_gt_box, batch_dims=1)
    expanded_gt_boxes = tf.where(tf.expand_dims(pos_cond, -1), gt_boxes_map, tf.zeros_like(gt_boxes_map))
    bbox_deltas = bbox_utils.abs2prop(anchors, expanded_gt_boxes) / variances
    #
    gt_labels_map = tf.gather(gt_labels, max_indices_each_gt_box, batch_dims=1)
    expanded_gt_labels = tf.where(pos_cond, gt_labels_map, tf.zeros_like(gt_labels_map))
    bbox_labels = tf.one_hot(expanded_gt_labels, total_labels)
    #
    if debug:
        pos = tf.argmax(pos_cond[0, :])
        print("==========")
        print(f"anchor => {anchors.shape}")
        print(f"anchor => {anchors[0, :]}")
        print(f"gt bx => {gt_boxes.shape}")
        print(f"gt bx => {gt_boxes[0, :, :]}")
        print(f"positive cond => {pos_cond.shape}")
        print(f"positive cond => {tf.argmax(pos_cond[0, :])}")
        print(f"iou map => {iou_map.shape}")
        print(f"iou map of idx {pos} => {iou_map[0, pos, :]}")
        print(f"anchors of idx {pos} => {anchors[pos, :]}")
        print(f"max indices => {max_indices_each_gt_box.shape}")
        print(f"max indices of idx {pos} => {max_indices_each_gt_box[0, pos]}")
        print(f"merged iou map => {merged_iou_map.shape}")
        print(f"merged iou map of idx {pos} => {merged_iou_map[0, pos]}")
        print(f"gt bx map => {gt_boxes_map.shape}")
        print(f"gt bx map of idx {pos} => {gt_boxes_map[0, pos, :]}")
        print(f"expanded gt bx=> {expanded_gt_boxes.shape}")
        print(f"expanded gt bx of idx {pos} => {expanded_gt_boxes[0, pos, :]}")
        print(f"gt label map => {gt_labels_map.shape}")
        print(f"gt label map of idx {pos} => {gt_labels_map[0, pos]}")
        print(f"expanded labels => {expanded_gt_labels.shape}")
        print(f"expanded labels of idx {pos} => {expanded_gt_labels[0, pos]}") 
        print(f"anchor of idx {pos} => {anchors[pos, :]}")
        print(f"bbox deltas => {bbox_deltas.shape}")
        print(f"bbox deltas of idx {pos} => {bbox_deltas[0, pos, :]}")
        print(f"bbox labels  => {bbox_labels.shape}")
        print(f"bbox labels of idx {pos} => {bbox_labels[0, pos, :]}")
    return bbox_deltas, bbox_labels