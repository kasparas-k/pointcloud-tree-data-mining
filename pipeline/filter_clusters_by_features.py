from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import shutil

from tqdm import tqdm
import pandas as pd


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('in_root', help='root path containing las point cloud and npy raster subdirs')
    parser.add_argument('out_root', help='root path to write individual tree point cloud result subdir and result summary')
    parser.add_argument('features_json_path')
    parser.add_argument('--min_extent', nargs='?', default=3.0, type=float, help='minimum extent (length units) for a candidate tree cluster')
    parser.add_argument('--max_extent', nargs='?', default=16.0, type=float, help='maximum extent (length units) for a candidate tree cluster')
    parser.add_argument('--max_aspect', nargs='?', default=1.4, type=float, help='maximum bounding box aspect ratio for a candidate tree cluster')
    parser.add_argument('--aspect_n_rotate', nargs='?', default=6, type=int, help='when checking maximum aspect ratio, how many rotation steps from 0 to π radians')
    parser.add_argument('--min_height', nargs='?', default=5.0, type=float, help='minimum tree height (length units)')
    parser.add_argument('--min_raster_fill', nargs='?', default=0.35, type=float, help='minimum ratio of tree are and total bounding box raster area')
    parser.add_argument('--max_width_to_height', nargs='?', default=1.0, type=float, help='maximum ratio between the tree width and heiht')
    parser.add_argument('--move', action='store_true', help='Set this flag to move original cluster files instead of copying them to the destination directory')
    args = parser.parse_args()

    in_root = Path(args.in_root).resolve()
    out_root = Path(args.out_root).resolve()
    features_json_path = args.features_json_path

    features = pd.read_json(features_json_path)
    checks = [
        lambda x: x['min_bbox_extent'] > args.min_extent,
        lambda x: x['max_bbox_extent'] < args.max_extent,
        lambda x: x['max_bbox_aspect_ratio'] < args.max_aspect,
        lambda x: x['height'] > args.min_height,
        lambda x: x['raster_area_fill_ratio'] > args.min_raster_fill,
        lambda x: x['distance_to_closest_marked_tree'] <  x['distance_centroid_to_furthest_cluster_point'],
        lambda x: x['max_height_to_width_ratio'] < args.max_width_to_height,
    ]
    for check in checks:
        features = features[check(features)]

    out_root.mkdir(exist_ok=True, parents=True)
    features.to_json(out_root / 'filtered_features.json')

    with ThreadPoolExecutor() as executor:
        tasks = []
        for pc_relp in features.index.values:
            out_path = out_root / pc_relp
            out_path.parent.mkdir(exist_ok=True, parents=True)
            file_operation = shutil.copy if not args.move else shutil.move
            tasks.append(executor.submit(shutil.copy, in_root / pc_relp, out_path))
        for t in tqdm(as_completed(tasks), total=len(tasks), desc='Copying filtered trees...'):
            t.result()
