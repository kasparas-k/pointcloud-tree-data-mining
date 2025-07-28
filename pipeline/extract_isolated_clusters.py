from argparse import ArgumentParser
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import geojson
import laspy
import numpy as np
import pyproj
from scipy.signal import correlate2d
import sklearn.cluster
from tqdm import tqdm

from fec import FEC


CRS = {
    'Poland': 'EPSG:2180',
    'Estonia': 'EPSG:3301',
    'Lithuania': 'EPSG:3346',
}


@dataclass
class TreeCandidate:
    pc: laspy.ScaleAwarePointRecord
    raster: np.ndarray


def get_object_centers(las: laspy.LasData, geojson_path: Path, country: str):
    """
    Given a point cloud and path to WGS84 projection geojson data, return object
    centers within the point cloud.

    Parameters:
        
    """
    xmax = np.max(las.x)
    xmin = np.min(las.x)
    ymax = np.max(las.y)
    ymin = np.min(las.y)

    with open(geojson_path) as f:
        geo = geojson.load(f)
    
    pointcloud_crs = CRS[country]
    geojson_crs = 'EPSG:4326'
    coordinate_transformer = pyproj.Transformer.from_crs(geojson_crs, pointcloud_crs, always_xy=True)
    
    centers = [
        coordinate_transformer.transform(*feat['geometry']['coordinates'])
        for feat in geo.features
    ]
    centers = [
        c for c in centers
        if (c[0] < xmax) and (c[0] > xmin)
        and (c[1] < ymax) and (c[1] > ymin)
    ]
    centers = np.array(centers)
    return centers


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


def get_isolated_clusters(
    clusters: dict[int, laspy.ScaleAwarePointRecord],
    global_binary_raster: np.ndarray,
    global_grid_offsets: np.ndarray,
    n_pad: int = 2,
    grid_size_m: float = 0.2,
    header: Optional[laspy.LasHeader] = None,
) -> dict[int, TreeCandidate]:
    """
    Given a dictionary of cluster points and the cluster ids, return clusters that are isolated

    Parameters:
        clusters: dict[int, laspy.ScaleAwarePointRecord]
            Dictionary of cluster ids and their point records
        global_binary_raster: np.ndarray
            Binary raster of the original point cloud the clusters originate from
        global_grid_offsets: np.ndarray
            x and y offsets for the raster pixel indices
        n_pad: int = 2
            Number of pixels to pad individual cluster rasters
        grid_size_m: float = 0.2,
            Grid size (meters) of the raster cell
        header: laspy.LasHeader
            Header for turning the extracted cluster's points into a point cloud

    Returns:
        isolated_clusters: dict[int, laspy.ScaleAwarePointRecord]
            Dictionary of ids and points of clusters that have no other cluster at the boundary
    """
    if header is None:
        raise ValueError('Did not provide LAS header')
    isolated_clusters = {} 
    for c_id, cluster in clusters.items():
        cluster_xy = np.hstack([cluster.x[:, np.newaxis], cluster.y[:, np.newaxis]])
        cluster_raster, gridded_points, _ = get_raster(cluster_xy, grid_size_m, None, n_pad=n_pad)
        gridded_points_offset = gridded_points - global_grid_offsets
    
        window_x1, window_y1 = np.clip(np.min(gridded_points_offset, axis=0) - n_pad, a_min=0, a_max=None)
        window_x2, window_y2 = np.max(gridded_points_offset, axis=0) + 1 + n_pad
    
        roi = global_binary_raster[window_x1:window_x2, window_y1:window_y2]
    
        # if the ROI shape is different, means the cluster touches the point cloud edge
        if roi.shape != cluster_raster.shape:
            continue
        
        cluster_outline = correlate2d(cluster_raster, np.ones((3, 3)), boundary='fill')
        cluster_outline = cluster_outline[1:-1, 1:-1]
        cluster_outline[cluster_raster == 1] = 0
        cluster_outline[cluster_outline > 1] = 1
    
        outline_overlap = np.zeros_like(roi)
        outline_overlap[(cluster_outline == 1) & (roi == 1)] = 1
        if np.sum(outline_overlap) == 0:
            las_cluster = laspy.LasData(header=header, points=cluster)
            isolated_clusters[c_id] = TreeCandidate(
                pc=las_cluster,
                raster=cluster_raster
            )

    return isolated_clusters


def extract_trees_in_point_cloud(
    las_path: Path,
    tree_class: list[int],
    country: str,
    tree_center_geojson: Path,
    fec_tolerance: float = 0.5,
    fec_max_n: int = 50,
    fec_min_component_size: int = 50,
    raster_grid_size: float = 0.2,
    use_meanshift: bool = False,
) -> dict[int, TreeCandidate]:
    try:
        las = laspy.read(las_path)
    except:
        print('Could not read ', las_path)
        return {}
    las.points = las.points[np.isin(las.classification, tree_class)]
    if len(las.points) == 0:
        return {}

    tree_centers = get_object_centers(
        las=las,
        geojson_path=tree_center_geojson,
        country=country
    )
    if tree_centers.size == 0:
        return {}

    if not use_meanshift:
        xyz = np.hstack([las.xyz[:, :2], np.zeros_like(las.x[:, np.newaxis])])
        xyz = xyz - np.min(xyz, axis=0)
        xyz[:, -1] = 1
        seg = FEC(xyz, min_component_size=fec_min_component_size, tolerance=fec_tolerance, max_n=fec_max_n)
    else:
        try:
            _, seg = sklearn.cluster.mean_shift(
                las.xyz[:, :2],
                bandwidth=0.6,
                seeds=tree_centers,
                cluster_all=True,
                n_jobs=-1
            )
        except ValueError:
            return {}

    cluster_to_point_idx = defaultdict(list)
    for i, c_id in enumerate(seg):
        cluster_to_point_idx[c_id].append(i)
    cluster_to_point_idx = {k: np.array(v) for k, v in cluster_to_point_idx.items()}
    clusters = {}
    for c_id, p_ids in cluster_to_point_idx.items():
        clusters[c_id] = las.points[p_ids]
    
    binary_raster, gridded_points, grid_offsets = get_raster(las.xyz[:, :2], raster_grid_size)
    isolated_clusters = get_isolated_clusters(
        clusters=clusters,
        global_binary_raster=binary_raster,
        global_grid_offsets=grid_offsets,
        grid_size_m=raster_grid_size,
        header=las.header,
    )
    
    return isolated_clusters


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('in_root', type=Path, help='root path of las/laz files')
    parser.add_argument('out_root', type=Path, help='root path to write individual tree point clouds')
    parser.add_argument('--country', nargs='?', required=True, help='point cloud origin country, used to project coordinates to WGS84')
    parser.add_argument('--tree_center_geojson', nargs='?', default=None, type=Path, help='path to the geojson file containing tree coordinates')
    parser.add_argument('--tree_class', nargs='+', type=int, required=True, help='one or more class ids that belong to tree class')
    parser.add_argument('--fec_tolerance', nargs='?', type=int, default=0.5, help='in FEC clustering, length of a sphere\'s radius defining the point\'s local neighborhood')
    parser.add_argument('--fec_max_n', nargs='?', type=int, default=50, help='in FEC clustering, maximum number of neighborhood points returned in nearest neighbor query')
    parser.add_argument('--fec_min_component_size', nargs='?', type=int, default=50, help='...')
    parser.add_argument('--raster_grid_size', nargs='?', type=float, default=0.2, help='Side length in length units of each raster pixel')
    parser.add_argument('--use_meanshift', action='store_true', help='Flag for using meanshift instead of FEC')
    args = parser.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)
    use_meanshift = args.use_meanshift

    pc_root = out_root / 'pc'
    raster_root = out_root / 'raster'
    log_root = out_root / 'log'

    las_files = list(args.in_root.rglob('*.laz')) + list(args.in_root.rglob('*.las')) 
    # for las_path in tqdm(las_files, desc='Extracting trees from point cloud...'):
    for las_path in tqdm(las_files):
        isolated_clusters = extract_trees_in_point_cloud(
            las_path=las_path,
            tree_class=args.tree_class,
            country=args.country,
            tree_center_geojson=args.tree_center_geojson,
            fec_tolerance=args.fec_tolerance,
            fec_max_n=args.fec_max_n,
            fec_min_component_size=args.fec_min_component_size,
            raster_grid_size=args.raster_grid_size,
            use_meanshift=use_meanshift,
        )

        with ThreadPoolExecutor() as executor:
            tasks = []
            for i, tree_candidate in isolated_clusters.items():
                out_relpath = las_path.relative_to(in_root).parent / las_path.stem

                out_las_path = pc_root / out_relpath / f'{i:06}.laz'
                out_las_path.parent.mkdir(exist_ok=True, parents=True)

                out_raster_path = raster_root / out_relpath / f'{i:06}.npy'
                out_raster_path.parent.mkdir(exist_ok=True, parents=True)
                tasks.append(executor.submit(tree_candidate.pc.write, out_las_path))
                tasks.append(executor.submit(np.save, out_raster_path, tree_candidate.raster))

            if tasks:
                for t in as_completed(tasks):
                    t.exception()
