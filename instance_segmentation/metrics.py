import numpy as np


EPS = np.finfo(float).eps  # machine epsilon for divisions


def get_instance_indices(las, attr='treeID'):
    attr_values = getattr(las, attr)
    ids = np.unique(attr_values)
    instance_idxs = [
        np.argwhere(attr_values == idx).flatten() for idx in ids
    ]
    return instance_idxs


def get_iou(a, b):
    intersection = np.intersect1d(a, b)
    union = np.union1d(a, b)
    return intersection.shape[0] / (union.shape[0] + EPS)


def get_iou_matrix(gt_ids, pred_ids):
    ious = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float32)
    for i, gt in enumerate(gt_ids):
        for j, pred in enumerate(pred_ids):
            ious[i, j] = get_iou(gt, pred)
    return ious


def coverage(iou_matrix):
    return np.mean(np.max(iou_matrix, axis=1))


def wcoverage(gt_idx, iou_matrix):
    total_gt_points = 0
    total_iou = 0
    best_pred_matches = np.argmax(iou_matrix, axis=1)
    for i, best_pred_idx in enumerate(best_pred_matches):
        instance_size = len(gt_idx[i])
        total_iou += instance_size * iou_matrix[i, best_pred_idx]
        total_gt_points += instance_size
    return total_iou / (total_gt_points + EPS)


def get_tp_fp_fn_matches(iou_matrix, threshold=0.5):
    tp = set()
    tp_pred = set()
    all_matches = []
    for gt_id, gt_matches in enumerate(iou_matrix):
        # over-threshold indices in order of decreasing iou
        best_matches = np.argsort(gt_matches)[np.sort(gt_matches) >= threshold][::-1]
        for bm in best_matches:
            if int(bm) not in tp_pred:
                tp.add(gt_id)
                tp_pred.add(int(bm))
                all_matches.append(dict(gt=gt_id, pred=int(bm), iou=iou_matrix[gt_id, bm]))
                break

    all_gt_idx = set(i for i in range(iou_matrix.shape[0]))
    all_pred_idx = set(i for i in range(iou_matrix.shape[1]))
    
    fn = all_gt_idx - tp
    fp = all_pred_idx - tp_pred
    return tp, fp, fn, all_matches


def prec_rec_f1(iou_matrix, threshold=0.5):
    tp, fp, fn, _ = tp, fp, fn, matches = get_tp_fp_fn_matches(iou_matrix, threshold)
    tp = len(tp)
    fp = len(fp)
    fn = len(fn)

    prec = tp / (tp + fp + EPS)
    rec = tp / (tp + fn + EPS)
    f1 = (2 * prec * rec) / (prec + rec + EPS)
    return prec, rec, f1


def evaluate(gt_idx: list[np.ndarray], pred_idx: list[np.ndarray], iou_threshold: float = 0.5) -> dict[str, float]:
    iou_matrix = get_iou_matrix(gt_idx, pred_idx)

    _prec, _rec, _f1 = prec_rec_f1(iou_matrix, iou_threshold)
    _cov =  coverage(iou_matrix)
    _wcov = wcoverage(gt_idx, iou_matrix)
    
    return dict(
        precision=float(_prec),
        recall=float(_rec),
        f1=float(_f1),
        coverage=float(_cov),
        wcoverage=float(_wcov),
    )
