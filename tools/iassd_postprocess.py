import numpy as np


def sigmoid(array):
    return 1.0 / (1.0 + np.exp(-array.astype(np.float64)))


def raw_outputs_to_candidates(outputs):
    cls_preds = outputs['batch_cls_preds']
    box_preds = outputs['batch_box_preds']
    if cls_preds.ndim == 3:
        cls_preds = cls_preds[0]
    if box_preds.ndim == 3:
        box_preds = box_preds[0]

    cls_scores = sigmoid(cls_preds)
    labels = np.argmax(cls_scores, axis=-1).astype(np.int64) + 1
    scores = np.max(cls_scores, axis=-1).astype(np.float32)
    return {
        'pred_boxes': box_preds.astype(np.float32),
        'pred_scores': scores,
        'pred_labels': labels,
    }


def boxes_to_aligned_bev(boxes):
    if boxes.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)

    centers = boxes[:, 0:2].astype(np.float32)
    sizes = boxes[:, 3:5].astype(np.float32)
    headings = boxes[:, 6].astype(np.float32)
    use_dx_dy = np.abs(np.cos(headings)) >= np.abs(np.sin(headings))
    aligned_sizes = sizes.copy()
    aligned_sizes[~use_dx_dy] = aligned_sizes[~use_dx_dy][:, ::-1]
    half_sizes = aligned_sizes * 0.5
    return np.concatenate([centers - half_sizes, centers + half_sizes], axis=1)


def aligned_bev_iou(reference_box, target_boxes):
    if target_boxes.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)

    lt = np.maximum(reference_box[:2], target_boxes[:, :2])
    rb = np.minimum(reference_box[2:], target_boxes[:, 2:])
    wh = np.maximum(rb - lt, 0.0)
    inter = wh[:, 0] * wh[:, 1]
    reference_area = max((reference_box[2] - reference_box[0]) * (reference_box[3] - reference_box[1]), 0.0)
    target_area = np.maximum(target_boxes[:, 2] - target_boxes[:, 0], 0.0) * np.maximum(
        target_boxes[:, 3] - target_boxes[:, 1],
        0.0,
    )
    union = reference_area + target_area - inter
    return inter / np.maximum(union, 1e-8)


def aligned_bev_nms(boxes, scores, nms_thresh, pre_maxsize, post_maxsize):
    if scores.shape[0] == 0:
        return np.asarray([], dtype=np.int64)

    order = np.argsort(-scores, kind='mergesort')
    if pre_maxsize is not None:
        order = order[:pre_maxsize]

    bev_boxes = boxes_to_aligned_bev(boxes[order, :7])
    keep = []
    candidate_order = np.arange(order.shape[0], dtype=np.int64)
    while candidate_order.size > 0 and len(keep) < post_maxsize:
        current = candidate_order[0]
        keep.append(current)
        if candidate_order.size == 1:
            break
        ious = aligned_bev_iou(bev_boxes[current], bev_boxes[candidate_order[1:]])
        candidate_order = candidate_order[1:][ious <= nms_thresh]

    return order[np.asarray(keep, dtype=np.int64)]


def postprocess_outputs(outputs, score_thresh=0.1, nms_thresh=0.01, nms_pre_maxsize=4096, nms_post_maxsize=500):
    candidates = raw_outputs_to_candidates(outputs)
    boxes = candidates['pred_boxes']
    scores = candidates['pred_scores']
    labels = candidates['pred_labels']

    if score_thresh is not None:
        score_mask = scores >= float(score_thresh)
        original_indices = np.nonzero(score_mask)[0]
        filtered_boxes = boxes[score_mask]
        filtered_scores = scores[score_mask]
    else:
        original_indices = np.arange(scores.shape[0], dtype=np.int64)
        filtered_boxes = boxes
        filtered_scores = scores

    selected = aligned_bev_nms(
        filtered_boxes,
        filtered_scores,
        float(nms_thresh),
        int(nms_pre_maxsize),
        int(nms_post_maxsize),
    )
    selected = original_indices[selected]
    return {
        'pred_boxes': boxes[selected].astype(np.float32),
        'pred_scores': scores[selected].astype(np.float32),
        'pred_labels': labels[selected].astype(np.int64),
    }
