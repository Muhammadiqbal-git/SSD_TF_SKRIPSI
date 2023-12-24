import tensorflow as tf
import numpy as np
from utils import bbox_utils

def init_stats(labels):
    stats = {}
    for i, label in enumerate(labels):
        if i == 0:
            stats[i] = {
                "fn": []
            }
        else:
            stats[i] = {
                "label": label,
                "total": 0,
                "tp": [],
                "fp": [],
                "fn": 0,
                "scores": [],
            }
    return stats

def update_stats(pred_bboxes, pred_labels, pred_scores, gt_boxes, gt_labels, stats):
    # print("pred bbox shape {}".format(pred_bboxes.shape))
    # print("gt bbox shape {}".format(gt_boxes.shape))
    # print("gt labels shape {}".format(gt_labels.shape))
    # print("pred labels shape {}".format(pred_labels.shape))
    # print("pred scores shape {}".format(pred_scores.shape))
    iou_map = bbox_utils.compute_iou(pred_bboxes, gt_boxes)
    merged_iou_map = tf.reduce_max(iou_map, axis=-1)
    max_indices_each_gt = tf.argmax(iou_map, axis=-1, output_type=tf.int32)
    scores_s = tf.where(tf.greater(merged_iou_map, 0), pred_scores, tf.zeros_like(pred_scores))
    sorted_idx_score = tf.argsort(scores_s, direction="DESCENDING")
    # print("sorted iou {}".format(sorted_idxs.shape))
    # print("sorted iou {}".format(sorted_idxs[:, :5]))
    # print("scores {}".format(scores_s[:, :5]))
    # print("scores {}".format(sorted_idx_score.shape))
    # print("scores {}".format(sorted_idx_score[:, :5]))
    # print("gt labels {}".format(gt_labels))
    # print("gt labels {}".format(tf.reshape(gt_labels, (-1,))))
    #
    count_holder = tf.unique_with_counts(tf.reshape(gt_labels, (-1,)))
    total_gtbox_batch = 0
    for i, gt_label in enumerate(count_holder[0]):
        if gt_label == -1:
            continue
        gt_label = int(gt_label)
        stats[gt_label]["total"] += int(count_holder[2][i])
        stats[gt_label]["fn"] += count_holder[2][i]
        total_gtbox_batch += count_holder[2][i]
        print("total {}".format(stats[gt_label]["total"]))
    for batch_idx, m in enumerate(merged_iou_map):
        true_labels = []
        for i, sorted_idx in enumerate(sorted_idx_score[batch_idx]):
            pred_label = pred_labels[batch_idx, sorted_idx]
            if pred_label == 0:
                continue
            #
            iou = merged_iou_map[batch_idx, sorted_idx]
            pred_label = int(pred_label)
            gt_idx = max_indices_each_gt[batch_idx, sorted_idx]
            gt_label = int(gt_labels[batch_idx, gt_idx])

            score = pred_scores[batch_idx, sorted_idx]
            stats[pred_label]["scores"].append(score)
            stats[pred_label]["tp"].append(0)
            stats[pred_label]["fp"].append(0)
            if iou >= 0.5 and pred_label == gt_label and gt_idx not in true_labels:
                stats[pred_label]["tp"][-1] = 1
                true_labels.append(gt_idx)
                stats[pred_label]["fn"] -= 1
            else:
                stats[pred_label]["fp"][-1] = 1
            #
        #
    #
    return stats

def calculate_ap(recall, precision):
    ap = 0
    pre_r = 0.0
    for p in range(len(precision)-1, 0, -1):
        precision[p-1] = np.maximum(precision[p], precision[p-1])
    for idx, r in enumerate(recall):
        if pre_r != r:
            ap += precision[idx] * (r-pre_r)
            pre_r = r
    return ap

def calculate_mAP(stats):
    aps = []
    for label in stats:
        if label == 0:
            continue
        label_stats = stats[label]
        tp = np.array(label_stats["tp"])
        fp = np.array(label_stats["fp"])
        scores = np.array(label_stats["scores"])
        ids = np.argsort(-scores)
        total = label_stats["total"]
        accumulated_tp = np.cumsum(tp[ids])
        accumulated_fp = np.cumsum(fp[ids])
        recall = accumulated_tp / total
        precision = accumulated_tp / (accumulated_fp + accumulated_tp)
        print("tplen {}".format(len(tp)))
        print("tp {}".format(np.sum(tp)))
        print("fn {}".format(label_stats["fn"]))
        print("fplen {}".format(len(fp)))
        print("fp {}".format(np.sum(fp)))
        print("scores {}".format(scores[ids]))
        print("tp {}".format(accumulated_tp))
        print("fp {}".format(accumulated_fp))
        ap = calculate_ap(recall, precision)
        stats[label]["recall"] = recall
        stats[label]["precision"] = precision
        stats[label]["AP"] = ap
        print("precision {}".format(stats[label]["precision"]))
        print("RECALL {}".format(stats[label]["recall"]))
        aps.append(ap)
    mAP = np.mean(aps)
    return stats, mAP

def evaluate_predictions(dataset, pred_bboxes, pred_labels, pred_scores, labels, batch_size):
    stats = init_stats(labels)
    for batch_idx, image_data in enumerate(dataset):
        imgs, gt_boxes, gt_labels = image_data
        start = batch_idx * batch_size
        end = start + batch_size
        batch_bboxes, batch_labels, batch_scores = pred_bboxes[start:end], pred_labels[start:end], pred_scores[start:end]
        stats = update_stats(batch_bboxes, batch_labels, batch_scores, gt_boxes, gt_labels, stats)

    stats, mAP = calculate_mAP(stats)
    print("mAP: {}".format(float(mAP)))
    return stats