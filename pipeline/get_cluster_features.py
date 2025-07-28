from collections import defaultdict
from argparse import ArgumentParser
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import geojson
import laspy
import numpy as np
import pandas as pd
import pyproj
from tqdm import tqdm


CRS = {
    'Poland': 'EPSG:2180',
    'Estonia': 'EPSG:3301',
    'Lithuania': 'EPSG:3346',
}


LAZRS_SINGLE_THREADED = laspy.compression.LazBackend(1)
def readLAS(laspath: Path) -> laspy.LasData:
    """
    Read a las/laz file with a single thread.
    """
    with open(laspath, 'rb') as f:
        reader = laspy.LasReader(f, laz_backend=LAZRS_SINGLE_THREADED)
        return reader.read()


def get_object_centers(las, geojson_path, country) -> tuple[np.ndarray, np.ndarray]:
    with open(geojson_path) as f:
        geo = geojson.load(f)
    
    pointcloud_crs = CRS[country]
    geojson_crs = 'EPSG:4326'
    coordinate_transformer = pyproj.Transformer.from_crs(geojson_crs, pointcloud_crs, always_xy=True)
    
    centers_wgs = [
        feat['geometry']['coordinates']
        for feat in geo.features
    ]
    centers_transformed = [
        coordinate_transformer.transform(*center)
        for center in centers_wgs
    ]

    if las is not None:
        xmax = np.max(las.x)
        xmin = np.min(las.x)
        ymax = np.max(las.y)
        ymin = np.min(las.y)
        centers_transformed = [
            c for c in centers_transformed
            if (c[0] < xmax) and (c[0] > xmin)
            and (c[1] < ymax) and (c[1] > ymin)
        ]

    return np.array(centers_wgs), np.array(centers_transformed)


def rotate(xy: np.ndarray, theta: float, center: tuple[float, float]) -> np.ndarray:
    """
    Rotation around an axis, perpendicular to the XY plane, centered on object
    
    Parameters:
        xy: np.ndarray
            xy values of the point cloud, shaped like (n, 2)
        theta: float
            rotation angle in radians
        center: tuple[float, float]
            x and y coordinates of rotation center
    Returns:
        np.ndarray
            rotated xyz values

    Reference:
        https://www.euclideanspace.com/maths/geometry/affine/aroundPoint/matrix2d/
    """
    rot_mat = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ])
    x, y = center
    rot_mat = np.array([
        [rot_mat[0,0], rot_mat[0,1], x - rot_mat[0,0]*x - rot_mat[0,1]*y],
        [rot_mat[1,0], rot_mat[1,1], y - rot_mat[1,0]*x - rot_mat[1,1]*y],
        [0, 0, 1]
    ])
    xy_temp = np.hstack([xy, np.ones((len(xy), 1))])
    return np.dot(rot_mat, xy_temp.T).T[:, :2]


def get_raster(
    xy: np.ndarray,
    grid_size_m: float,
    grid_offsets: Optional[np.ndarray] = None,
    n_pad: int = 0
) -> tuple[np.ndarray]:
    """
    Get a binary raster, where 1 means there is at least 1 point in the pixel.

    Parameters:
        xy: np.ndarray
            x and y coordinates of the point cloud's points
        grid_size_m: float
            Raster pixel dimensions in meters
        grid_offsets: Optional[np.ndarray]
            Offset for the raster indices, if not provided the point cloud
        n_pad: int = 0
            Number of pixels to pad the raster edges

    Return:
        binary_raster: np.ndarray
            rastered point cloud
        gridded_points: np.ndarray
            point cloud points with their coordinates at the top-left edge of the pixel
        grid_offsets: np.ndarray
            x and y offsets for the raster indices, by default it's the minimum gridded
            coordinate on that axis
    """
    gridded_points = np.round(xy / grid_size_m).astype(np.int32)
    if grid_offsets is None:
        grid_offsets = np.min(gridded_points, axis=0)
    offset_points = gridded_points - grid_offsets
    raster_dimensions = np.max(offset_points, axis=0) + 1
    occupied_raster_pixels = set([(x, y) for x, y in offset_points])
    occupied_rows, occupied_cols = zip(*occupied_raster_pixels)
    
    binary_raster = np.zeros(raster_dimensions)
    binary_raster[occupied_rows, occupied_cols] = 1

    if n_pad > 0:
        binary_raster = np.pad(binary_raster, pad_width=n_pad, mode='constant', constant_values=0)

    return binary_raster, gridded_points, grid_offsets


def get_bbox_min_max_extent(pc: laspy.ScaleAwarePointRecord) -> np.ndarray:
    """
    Check whether the point cloud's bouding box's extent in meters fits within the given bounds

    Parameters:
        pc: laspy.ScaleAwarePointRecord
            Points of the point cloud to check

    Returns:
        bool
            True if the box fits within range
    """
    xy = np.hstack([pc.x[:, np.newaxis], pc.y[:, np.newaxis]])
    ext = np.ptp(xy, axis=0)
    return ext.min(), ext.max()


def get_biggest_aspect_ratio(pc: laspy.ScaleAwarePointRecord, n_rotate: int = 6, z_extent: bool = False) -> float:
    """
    Get the biggest xy plane bounding box aspect ratio and biggest y / z ratio

    Parameters:
        pc: laspy.ScaleAwarePointRecord
            Points of the point cloud to check
        n_rotate: int = 6
            How many rotation steps from 0 to π radians

    Returns:
        bool
            True if none of the rotated point cloud's bounding boxes exceed the given aspect ratio
    """
    xyz = np.hstack([pc.x[:, np.newaxis], pc.y[:, np.newaxis], pc.z[:, np.newaxis]])
    xy = xyz[:, :2]
    ext = np.ptp(xy, axis=0)

    x_c = (np.max(pc.x) + np.min(pc.x)) / 2
    y_c = (np.max(pc.y) + np.min(pc.y)) / 2

    ratios_horizontal = []
    ratios_vertical = []
    height = get_height(pc, z_extent=z_extent)
    for a in np.linspace(0, np.pi, n_rotate)[1:]:
        xy_rotated = rotate(xy, a, (x_c, y_c))
        ext = np.ptp(xy_rotated, axis=0)
        ratios_horizontal += [
            ext[0] / ext[1],
            ext[1] / ext[0],
        ]
    
        ratios_vertical += [
            ext[1] / height
        ]

    return max(ratios_horizontal), max(ratios_vertical), height


def get_height(pc: laspy.ScaleAwarePointRecord, z_extent: bool) -> bool:
    """
    Check whether the candidate tree cluster meets height requirements.
    
    Parameters:
        pc: laspy.ScaleAwarePointRecord
            Points of the point cloud to check
        min_height: float = 5.0
            Minimum height for a cluster to be considered a tree

    Returns:
        bool
            True if the tree is taller than the minimum required height
    """
    if z_extent:
        return np.max(pc.z) - np.min(pc.z)
    return np.max(pc.z)


def get_raster_area_fill_ratio(binary_raster: np.ndarray) -> float:
    """
    Gets the ratio of the filled and all raster pixels.

    Parameters:
        binary_raster: np.ndarray
            raster containing 0 and 1 values, with 1 corresponding to tree pixels
    Returns:
        bool
            The ratio of tree and full raster area
    """
    area_fill_ratio = np.sum(binary_raster) / np.prod(binary_raster.shape)
    return area_fill_ratio


def get_distance_to_closest_tree(pc: laspy.ScaleAwarePointRecord, tree_centers_local: np.ndarray, tree_centers_wgs: np.ndarray) -> float:
    """
    Check whether there is a corresponding prior tree location for the cluster.

    Parameters:
        pc: laspy.ScaleAwarePointRecord
            Points of the point cloud to check
        tree_centers: np.ndarray
            (n, d) shaped array with all known tree centers

    Returns:
        bool
            True if there is a known tree location that corresponds to the target cluster
    """
    
    xmax = np.max(pc.x)
    xmin = np.min(pc.x)
    ymax = np.max(pc.y)
    ymin = np.min(pc.y)

    xy = np.vstack([pc.x, pc.y]).T
    cluster_centroid = np.array([[np.mean(pc.x), np.mean(pc.y)]])
    max_centroid_dist = np.linalg.norm(xy - cluster_centroid, axis=1).max()
    cluster_centroid = cluster_centroid.squeeze()

    if tree_centers_local is None:
        return np.inf, max_centroid_dist, cluster_centroid, np.array([np.inf, np.inf]), np.array([np.inf, np.inf])

    tree_centers_mask = (
        (tree_centers_local[:, 0] < xmax)
        & (tree_centers_local[:, 0] > xmin)
        & (tree_centers_local[:, 1] < ymax)
        & (tree_centers_local[:, 1] > ymin)
    )
    
    tree_centers = tree_centers_local[tree_centers_mask]
    if tree_centers.size == 0:
        return np.inf, max_centroid_dist, cluster_centroid, np.array([np.inf, np.inf]), np.array([np.inf, np.inf])
    tree_dist = np.linalg.norm(tree_centers - cluster_centroid)
    closest_idx = np.argmin(tree_dist)
    closest_tree = tree_centers[closest_idx]
    closest_tree_wgs = tree_centers_wgs[tree_centers_mask][closest_idx]
    return tree_dist.min(), max_centroid_dist, cluster_centroid, closest_tree, closest_tree_wgs


def get_cluster_features(
    pc_and_raster: dict[str, Path],
    pc_root: Path,
    n_rotate: int,
    tree_centers_wgs: list,
    tree_centers_local: np.ndarray,
    raster_grid_size: float,
    z_extent: bool,
) -> dict[str, dict[str, float]]:
    """
    Calculate cluster features for filtering and tree matching

    Parameters:
        ...

    Returns:
        ...
    """
    pc_path = pc_root / pc_and_raster['pc']
    pc = readLAS(pc_path).points
    if 'raster' not in pc_and_raster:
        raster, _, _ = get_raster(np.vstack([pc.x, pc.y]).T, grid_size_m=raster_grid_size, n_pad=2)
    else:
        raster = np.load(pc_root / pc_and_raster['raster'])
    min_tree_dist, max_centroid_dist, cluster_centroid, closest_tree_local, closest_tree_wgs = (
        get_distance_to_closest_tree(pc, tree_centers_local, tree_centers_wgs)
    )
    min_ext, max_ext = get_bbox_min_max_extent(pc)
    biggest_xy, biggest_yz, height = get_biggest_aspect_ratio(pc, n_rotate, z_extent)
    return {
        str(pc_path.relative_to(pc_root)): dict(
            min_bbox_extent=min_ext,
            max_bbox_extent=max_ext,
            max_bbox_aspect_ratio=biggest_xy,
            max_height_to_width_ratio=biggest_yz,
            height=height,
            raster_area_fill_ratio=get_raster_area_fill_ratio(raster),
            distance_to_closest_marked_tree=min_tree_dist,
            distance_centroid_to_furthest_cluster_point=max_centroid_dist,
            cluster_centroid=cluster_centroid,
            closest_tree_coordinates=closest_tree_local,
            closest_tree_wgs=closest_tree_wgs,
        )
    }


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('in_root', help='root path containing las point cloud and npy raster subdirs')
    parser.add_argument('out_json_path', help='path to the output json of dataframe with cluster features')
    parser.add_argument('--aspect_n_rotate', nargs='?', default=6, type=int, help='when checking maximum aspect ratio, how many rotation steps from 0 to π radians')
    parser.add_argument('--country', nargs='?', default=None, help='point cloud origin country, used to project coordinates to WGS84')
    parser.add_argument('--tree_center_geojson', nargs='?', default=None, type=Path, help='path to the geojson file containing tree coordinates')
    parser.add_argument('--raster_grid_size', nargs='?', type=float, default=0.2, help='Side length in length units of each raster pixel')
    parser.add_argument('--nogeo', action='store_true', help='Give this flag to skip the closest object location matching')
    parser.add_argument('--z_extent', action='store_true', help='set this flag to calculate height based on z axis extent can be used when ground points or the full stem is present in individual tree point clouds)')
    parser.add_argument('--n_proc', nargs='?', default=None, type=int, help='Number of processor cores to use')
    args = parser.parse_args()

    in_root = Path(args.in_root)
    out_json_path = Path(args.out_json_path)
    out_json_path.parent.mkdir(exist_ok=True, parents=True)
    use_geodata = not args.nogeo

    if use_geodata:
        assert (args.country is not None) and (args.tree_center_geojson is not None), 'Must provide country and tree locations, unless --nogeo flag is given'

    pcs = [l.relative_to(in_root) for l in in_root.rglob('*.laz')]
    rasters = [r.relative_to(in_root) for r in in_root.rglob('*.npy')]
    pc_and_raster_paths = defaultdict(dict)
    for pc in pcs:
        pc_and_raster_paths[Path(*pc.parts[1:-1], pc.stem)]['pc'] = pc
    for raster in rasters:
        pc_and_raster_paths[Path(*raster.parts[1:-1], raster.stem)]['raster'] = raster

    if use_geodata:
        tree_centers_wgs, tree_centers_local = get_object_centers(None, args.tree_center_geojson, args.country)
    else:
        tree_centers_local = None
        tree_centers_wgs = None

    get_features_partial = partial(
        get_cluster_features,
        pc_root=in_root,
        n_rotate=args.aspect_n_rotate,
        tree_centers_local=tree_centers_local,
        tree_centers_wgs=tree_centers_wgs,
        raster_grid_size=args.raster_grid_size,
        z_extent=args.z_extent,
    )

    cluster_features = {}
    with Pool(processes=args.n_proc) as pool:
        imap = pool.imap(get_features_partial, pc_and_raster_paths.values())
        for res in tqdm(imap, total=len(pc_and_raster_paths)):
            cluster_features.update(res)

    pd.DataFrame(cluster_features).T.to_json(out_json_path) #, index_label='pc_relpath')
