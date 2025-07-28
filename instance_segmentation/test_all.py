from argparse import ArgumentParser
import json
from pathlib import Path

import laspy

from metrics import get_instance_indices, evaluate


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('in_root', type=Path)
    parser.add_argument('out_file', type=Path)
    args = parser.parse_args()

    in_root = Path(args.in_root).resolve()
    gts = list(in_root.rglob('gt.laz')) + list(in_root.rglob('gt.las'))
    eval_results = {}
    for gt_path in gts:
        eval_result = {}
        gt_idx = get_instance_indices(laspy.read(gt_path))
        res = list(gt_path.parent.rglob('*.laz')) + list(gt_path.parent.rglob('*.las'))
        for pred_path in res:
            if pred_path == gt_path:
                continue
            pred_las = laspy.read(pred_path)
            pred_idx = get_instance_indices(pred_las)
            
            # for methods that do not preserve original point cloud's point ordering,
            # ground trtuth ids need to be stored inside the result. We use a field 'gt_treeID'         
            if hasattr(pred_las, 'gt_treeID'):
                pred_gt_idx = get_instance_indices(pred_las, 'gt_treeID')
                eval_result[pred_path.stem] = evaluate(pred_gt_idx, pred_idx)
            else:    
                eval_result[pred_path.stem] = evaluate(gt_idx, pred_idx)
        eval_results[str(gt_path.parent.relative_to(in_root))] = eval_result

    with open(args.out_file, 'w') as f:
        json.dump(eval_results, f, indent=2)    
