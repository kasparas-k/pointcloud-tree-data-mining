from argparse import ArgumentParser
from typing import Optional
from pathlib import Path

import laspy
import numpy as np
import pandas as pd

from transform import perturb, rotate
from gaussian import get_gauss_to_gauss_transform


def match_trees(
        distribution_df: pd.DataFrame,
        individual_tree_feature_df: pd.DataFrame,
        height_tolerance: float = 2.0,
        width_tolerance: float = 0.5,
        xy_scale_adjust: float = 1.0,
        match_dimension_distributions: bool = False,
        match_gauss_to_range: bool = False,
        seed: int = 1,
) -> list[dict]:
    """
    """
    if seed is not None:
        np.random.seed(seed)

    if match_dimension_distributions:
        transform_h, _ = get_gauss_to_gauss_transform(
            np.array(distribution_df.height),
            np.array(individual_tree_feature_df.height),
            right_margin=height_tolerance,
            match_to_range=match_gauss_to_range,
        )

        transform_w, (scale_w, _) = get_gauss_to_gauss_transform(
            np.array(distribution_df.max_bbox_extent),
            np.array(individual_tree_feature_df.max_bbox_extent),
            right_margin=width_tolerance,
            match_to_range=match_gauss_to_range,
        )
    else:
        transform_h = lambda x: x
        transform_w = lambda x: x
        scale_w = 1.0

    offset_x = distribution_df.left.min()
    offset_y = distribution_df.bottom.min()

    placed_trees = []
    for _, tomatch in distribution_df.iterrows():
        candidates = individual_tree_feature_df[
            (individual_tree_feature_df.height > transform_h(tomatch.height)) 
            & (individual_tree_feature_df.height < (transform_h(tomatch.height) + height_tolerance))
            & (individual_tree_feature_df.max_bbox_extent > transform_w(tomatch.max_bbox_extent)) 
            & (individual_tree_feature_df.max_bbox_extent < (transform_w(tomatch.max_bbox_extent) + width_tolerance))
        ]
        if len(candidates) == 0:
            continue
        chosen_candidate = np.random.choice(candidates.index)

        position_scale = scale_w * xy_scale_adjust
        placed_trees.append(dict(
            pc_path=chosen_candidate,
            center=(
                position_scale * (tomatch.center_x - offset_x),
                position_scale * (tomatch.center_y - offset_y),
            )
        ))
    
    return placed_trees


def make_synthetic_scene(placed_trees: list[dict], tree_root: Path, rgb: bool = False, seed: int = 1) -> laspy.LasData:
    """
    """
    if seed is not None:
        np.random.seed(seed)

    attr_names = ['x', 'y', 'z', 'treeID']
    if rgb:
        attr_names.extend(['red', 'green', 'blue'])
    scene_attributes = {}

    for i, tree in enumerate(placed_trees):
        las = laspy.read(tree_root / tree['pc_path'])
        c_x, c_y = tree['center']
        las.x = (las.x - np.mean(las.x)) + c_x
        las.y = (las.y - np.mean(las.y)) + c_y

        las.xyz = rotate(las.xyz, rot_center=(c_x, c_y), flip=True)
        las.xyz = perturb(las.xyz, magnitude=0.2)
        
        las.add_extra_dim(laspy.ExtraBytesParams(
            name="treeID",
            type=np.int32,
        ))
        las.treeID = np.full(np.array(las.x).shape, i)

        for attr in attr_names:
            if attr not in scene_attributes:
                scene_attributes[attr] = getattr(las, attr)
            else:
                scene_attributes[attr] = np.concatenate([scene_attributes[attr], getattr(las, attr)])

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.add_extra_dim(laspy.ExtraBytesParams(name="treeID", type=np.int32))
    las_scene = laspy.LasData(header)
    for attr in attr_names:
        setattr(las_scene, attr, scene_attributes[attr])

    return las_scene


def pick_plot(df: pd.DataFrame, plot_id: str, bbox: Optional[list[float]] = None, bbox_is_absolute: bool = False):
    """
    """
    df_ret = df[df.geo_index == plot_id]
    left = df_ret.left.min()
    right = df_ret.right.max()
    top = df_ret.top.max()
    bottom = df_ret.bottom.min()

    if bbox is not None:
        x, y, w, h = bbox
    else:
        x, y, w, h = 0, 0, right-left, top-bottom

    if bbox_is_absolute:
        left = x
        top = y
    else:
        left = left + x
        top = top - y
    right = left + w
    bottom = top - h

    df_ret = df_ret[
        (left <= df_ret.left) & (df_ret.right < right)
        & (bottom <= df_ret.bottom) & (df_ret.top < top)
    ]

    return df_ret


def add_center_and_extent(df: pd.DataFrame) -> pd.DataFrame:
    columns = list(df.columns)
    columns.extend(['max_bbox_extent', 'center_x', 'center_y'])
    df['right'] = df['right'] - df['left'].min()
    df['left'] = df['left'] - df['left'].min()
    df['top'] = df['top'] - df['bottom'].min()
    df['bottom'] = df['bottom'] - df['bottom'].min()
    df['center_x'] = (df['right'] + df['left']) / 2
    df['center_y'] = (df['top'] + df['bottom']) / 2
    df['ext_x'] = df['right'] - df['left']
    df['ext_y'] = df['top'] - df['bottom']
    df['max_bbox_extent'] = df[['ext_x', 'ext_y']].max(axis=1)
    return df[columns]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('tree_root', type=Path, help='Root directory containing individual tree point clouds')
    parser.add_argument('tree_feature_json', help='Path to the feature json for the specified tree root directory')
    parser.add_argument('distribution_csv', help='Path to the csv file containing the target tree distribution')
    parser.add_argument('out_root', type=Path, help='Root to output scene point clouds. If plot_ids is not specified, all plots will be used and written to separate files.')
    parser.add_argument('--seed', type=int, default=1, nargs='?', help='Random number seed')
    parser.add_argument('--match_gauss', action='store_true', help='Set this flag to match gaussian distribution of the tree heights and widths to the input individual tree population')
    parser.add_argument('--plot_ids', nargs='*', help='If given, only output target plots from the given ID lists')
    parser.add_argument('--bbox', nargs='*', type=float, help='Bounding box (left_x, top_y, width, height) to make the scene. Relative to plot\'s top left corner by default.')
    parser.add_argument('--bbox_absolute', action='store_true', help='Set this flag to make bounding box coordinates absolute. Careful when using without specifying plot_ids.')
    parser.add_argument('--dh', type=float, default=2.0, help='Height tolerance for matching trees. The match interval becomes [height, height + dh)')
    parser.add_argument('--dw', type=float, default=0.5, help='Width tolerance for matching trees. The match interval becomes [width, width + dw)')
    parser.add_argument('--xy_scale', type=float, default=1.0, help='Multiply xy position by the xy_scale to change tree density while keeping other parameters the same.')
    parser.add_argument('--rgb', action='store_true', help='Set this flag to write rgb features (will lead to errors if input point clouds do not have color)')
    parser.add_argument('--match_gauss_to_range', action='store_true', help='If sthis flag is set, fit gaussian distributions to parameter range and middle, instead of fitting a gaussian to target distribution')
    args = parser.parse_args()
        
    # set random seed here, provide seed as None to all other functions not to re-seed it
    np.random.seed(args.seed)

    tree_df = pd.read_json(args.tree_feature_json)
    distribution_df = pd.read_csv(args.distribution_csv)
    plot_ids = args.plot_ids
    if plot_ids is None:
        plot_ids = distribution_df.geo_index.unique()

    for plot_id in plot_ids:
        distribution_df_subset = pick_plot(distribution_df, plot_id=plot_id, bbox=args.bbox, bbox_is_absolute=args.bbox_absolute)
        distribution_df_subset = add_center_and_extent(distribution_df_subset)
    
        matched_trees = match_trees(
            distribution_df_subset,
            tree_df,
            height_tolerance=args.dh,
            width_tolerance=args.dw,
            xy_scale_adjust=args.xy_scale,
            match_dimension_distributions=args.match_gauss,
            match_gauss_to_range=args.match_gauss_to_range,
            seed=None,
        )
        if len(matched_trees) == 0:
            print(f'No trees matched for {plot_id}')
            continue

        scene = make_synthetic_scene(placed_trees=matched_trees, tree_root=args.tree_root.resolve(), rgb=args.rgb, seed=None)
        out_path = args.out_root.resolve() / f'{plot_id}.laz'
        out_path.parent.mkdir(exist_ok=True, parents=True)
        scene.write(out_path)
