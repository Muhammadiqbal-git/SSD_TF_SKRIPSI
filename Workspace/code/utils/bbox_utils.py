import tensorflow as tf

def nms(pred_bboxes, pred_labels, **kwargs):
    """Apply non-maximum-suppresing (NMS)

    Args:
        pred_bboxes (tf.tensor): tensor of predicted bboxes.
                (batch_size, total_boxes, total_labels, [ymin, xmin, ymax, xmax])
        pred_labels (tf.tensor): tensor of predicted labels.
                (batch_size, total_boxes, total_labels)
    
    Returns:
        nms_boxes (type): (batch_size, max_detection, [ymin, xmin, ymax, xmax])
        nms_scores (type): (batch_size, max_detection)
        nms_classes (type): (batch_size, max_detection)
        valid_nms (type): valid detection are valid_nms[i] and nms_boxes[i] and nms_class[i]
                (batch_size)
    """
    return tf.image.combined_non_max_suppression(
        pred_bboxes,
        pred_labels,
        **kwargs
    )
    
def compute_iou(bboxes, gt_boxes, transpose_perm=[0, 2, 1]):
    """Calculating intersection over union (IOU)

    Args:
        bboxes (tf.tensor): tensor of bboxes
                (dynamic_dimension, [ymin, xmin, ymax, xmax])
        gt_boxes (tf.tensor): tensor of ground truth boxes
                (dynamic_dimension, [ymin, xmin, ymax, xmax])
        transpose_perm (list, optional): transpose perm order for 3d gt_boxes. Defaults to [0, 2, 1].

    
    Returns:
        iou_map (tf.tensor): tensor of iou score
                (dynamic_dimension, number_gt_boxes)
    """
    gt_rank = tf.rank(gt_boxes)
    gt_expand_axis = gt_rank - 2

    bbox_ymin, bbox_xmin, bbox_ymax, bbox_xmax = tf.split(bboxes, 4, axis=-1)
    gt_ymin, gt_xmin, gt_ymax, gt_xmax = tf.split(gt_boxes, 4, axis=-1)

    gt_area = tf.squeze((gt_ymax - gt_ymin) * (gt_xmax - gt_xmin), axis=-1)
    bbox_area = tf.squeze((bbox_ymax - bbox_ymin) * (gt_xmax - gt_xmin), axis=-1)

    x_top = tf.maximum(bbox_xmin, tf.transpose(gt_xmin, transpose_perm))
    y_top = tf.maximum(bbox_ymin, tf.transpose(gt_ymin, transpose_perm))
    x_bot = tf.minimum(bbox_xmax, tf.transpose(gt_xmax, transpose_perm))
    y_bot = tf.minimum(bbox_ymax, tf.transpose(gt_ymax, transpose_perm))

    i_area = tf.maximum(x_bot - x_top, 0) * tf.maximum(y_bot-y_top, 0)
    u_area = (tf.expand_dims(bbox_area, -1) + tf.expand_dims(gt_area, gt_expand_axis) - i_area)

    return i_area/u_area

def prop2abs(prior_boxes, deltas):
    """Calculating bounding boxes for given bounding box and delta values

    Args:
        prior_boxes (_type_): _description_
        deltas (_type_): _description_

    Returns:
        _type_: _description_
    """
    all_pbox_width = prior_boxes[..., 3] - prior_boxes[..., 1]
    all_pbox_height = prior_boxes[..., 2] - prior_boxes[..., 0]
    all_pbox_ctr_x = prior_boxes[..., 1] + 0.5 * all_pbox_width
    all_pbox_ctr_y = prior_boxes[..., 0] + 0.5 * all_pbox_height
    #
    all_bbox_width = tf.exp(deltas[..., 3]) * all_pbox_width
    all_bbox_height = tf.exp(deltas[..., 2]) * all_pbox_height
    all_bbox_ctr_x = (deltas[..., 1] * all_pbox_width) + all_pbox_ctr_x
    all_bbox_ctr_y = (deltas[..., 0] * all_pbox_height) + all_pbox_ctr_y
    #
    ymin = all_bbox_ctr_y - (0.5 * all_bbox_height)
    xmin = all_bbox_ctr_x - (0.5 * all_bbox_width)
    ymin = all_bbox_height + ymin
    xmin = all_bbox_width + xmin
    #
    return tf.stack([ymin, xmin, ymin, xmin], axis=-1)

def abs2prop(bboxes, gt_boxes):
    """Calculating bounding box proportional value (deltas) for given bounding box and ground truth boxes.
    Args:
        bboxes (tf.tensor): (total_bboxes, [y1, x1, y2, x2])
        gt_boxes (tf.tensor): (batch_size, total_bboxes, [y1, x1, y2, x2])

    Returns:
        final_deltas (tf.tensor): (batch_size, total_bboxes, [delta_y, delta_x, delta_h, delta_w])
    """
    bbox_width = bboxes[..., 3] - bboxes[..., 1]
    bbox_height = bboxes[..., 2] - bboxes[..., 0]
    bbox_ctr_x = bboxes[..., 1] + 0.5 * bbox_width
    bbox_ctr_y = bboxes[..., 0] + 0.5 * bbox_height
    #
    gt_width = gt_boxes[..., 3] - gt_boxes[..., 1]
    gt_height = gt_boxes[..., 2] - gt_boxes[..., 0]
    gt_ctr_x = gt_boxes[..., 1] + 0.5 * gt_width
    gt_ctr_y = gt_boxes[..., 0] + 0.5 * gt_height
    #
    bbox_width = tf.where(tf.equal(bbox_width, 0), 1e-3, bbox_width)
    bbox_height = tf.where(tf.equal(bbox_height, 0), 1e-3, bbox_height)
    delta_x = tf.where(tf.equal(gt_width, 0), tf.zeros_like(gt_width), tf.truediv((gt_ctr_x - bbox_ctr_x), bbox_width))
    delta_y = tf.where(tf.equal(gt_height, 0), tf.zeros_like(gt_height), tf.truediv((gt_ctr_y - bbox_ctr_y), bbox_height))
    delta_w = tf.where(tf.equal(gt_width, 0), tf.zeros_like(gt_width), tf.math.log(gt_width / bbox_width))
    delta_h = tf.where(tf.equal(gt_height, 0), tf.zeros_like(gt_height), tf.math.log(gt_height / bbox_height))
    #
    return tf.stack([delta_y, delta_x, delta_h, delta_w], axis=-1)

def scale_for_k_feature_map(k, m=6, scale_min=0.2, scale_max=0.9):
    """Calculating scale value for kth feature map based on original SSD paper.
    Args:
        k (scalar): kth feature map for scale calculation
        m (scalar): length of all using feature maps for detections. Default 6 for ssd300

    Returns:
        scale (scalar): calculated scale value for given kth
    """
    return scale_min + ((scale_max - scale_min) / (m - 1)) * (k - 1)

def generate_base_prior_boxes(aspect_ratios, feature_map_index, total_feature_map):
    """Generating top left prior boxes for given stride, height and width pairs of different aspect ratios.
    These prior boxes same with the anchors in Faster-RCNN.
    Args:
        aspect_ratios : for all feature map shapes + 1 for ratio 1
        feature_map_index : nth feature maps for scale calculation
        total_feature_map : length of all using feature map for detections, 6 for ssd300

    Returns:
        base_prior_boxes : (prior_box_count, [y1, x1, y2, x2])
    """
    current_scale = scale_for_k_feature_map(feature_map_index, m=total_feature_map)
    next_scale = scale_for_k_feature_map(feature_map_index + 1, m=total_feature_map)
    base_prior_boxes = []
    for aspect_ratio in aspect_ratios:
        height = current_scale / tf.sqrt(aspect_ratio)
        width = current_scale * tf.sqrt(aspect_ratio)
        base_prior_boxes.append([-height/2, -width/2, height/2, width/2])
    # 1 extra pair for ratio 1
    height = width = tf.sqrt(current_scale * next_scale)
    base_prior_boxes.append([-height/2, -width/2, height/2, width/2])
    return tf.cast(base_prior_boxes, dtype=tf.float32)

def generate_prior_boxes(feature_map_shapes, aspect_ratios):
    """Generating top left prior boxes for given stride, height and width pairs of different aspect ratios.
    These prior boxes same with the anchors in Faster-RCNN.
    Args:
        feature_map_shapes : for all feature map output size
        aspect_ratios : for all feature map shapes + 1 for ratio 1

    Returns:
        prior_boxes : (total_prior_boxes, [y1, x1, y2, x2])
            these values in normalized format between [0, 1]
    """
    prior_boxes = []
    for i, feature_map_shape in enumerate(feature_map_shapes):
        base_prior_boxes = generate_base_prior_boxes(aspect_ratios[i], i+1, len(feature_map_shapes))
        #
        stride = 1 / feature_map_shape
        grid_coords = tf.cast(tf.range(0, feature_map_shape) / feature_map_shape + stride / 2, dtype=tf.float32)
        grid_x, grid_y = tf.meshgrid(grid_coords, grid_coords)
        flat_grid_x, flat_grid_y = tf.reshape(grid_x, (-1, )), tf.reshape(grid_y, (-1, ))
        #
        grid_map = tf.stack([flat_grid_y, flat_grid_x, flat_grid_y, flat_grid_x], -1)
        #
        prior_boxes_for_feature_map = tf.reshape(base_prior_boxes, (1, -1, 4)) + tf.reshape(grid_map, (-1, 1, 4))
        prior_boxes_for_feature_map = tf.reshape(prior_boxes_for_feature_map, (-1, 4))
        #
        prior_boxes.append(prior_boxes_for_feature_map)
    prior_boxes = tf.concat(prior_boxes, axis=0)
    return tf.clip_by_value(prior_boxes, 0, 1)

def renormalize_bboxes_with_min_max(bboxes, min_max):
    """Renormalizing given bounding boxes to the new boundaries.
    r = (x - min) / (max - min)
    Args:
        bboxes : (total_bboxes, [y1, x1, y2, x2])
        min_max : ([y_min, x_min, y_max, x_max])
    """
    y_min, x_min, y_max, x_max = tf.split(min_max, 4)
    renomalized_bboxes = bboxes - tf.concat([y_min, x_min, y_min, x_min], -1)
    renomalized_bboxes /= tf.concat([y_max-y_min, x_max-x_min, y_max-y_min, x_max-x_min], -1)
    return tf.clip_by_value(renomalized_bboxes, 0, 1)

def normalize_bboxes(bboxes, height, width):
    """Normalizing bounding boxes.
    Args:
        bboxes : (batch_size, total_bboxes, [ymin, xmin, ymax, xmax])
        height : image height
        width : image width

    Returns:
        normalized_bboxes : (batch_size, total_bboxes, [ymin, xmin, ymax, xmax])
            in normalized form [0, 1]
    """
    ymin = bboxes[..., 0] / height
    xmin = bboxes[..., 1] / width
    ymax = bboxes[..., 2] / height
    xmax = bboxes[..., 3] / width
    return tf.stack([ymin, xmin, ymax, xmax], axis=-1)

def denormalize_bboxes(bboxes, height, width):
    """Denormalizing bounding boxes.
    Args:
        bboxes : (batch_size, total_bboxes, [ymin, xmin, ymax, xmax])
            in normalized form [0, 1]
        height : image height
        width : image width

    Returns:
        denormalized_bboxes : (batch_size, total_bboxes, [ymin, xmin, ymax, xmax])
    """
    ymin = bboxes[..., 0] * height
    xmin = bboxes[..., 1] * width
    ymax = bboxes[..., 2] * height
    xmax = bboxes[..., 3] * width
    return tf.round(tf.stack([ymin, xmin, ymax, xmax], axis=-1))