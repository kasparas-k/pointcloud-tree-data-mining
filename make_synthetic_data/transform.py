import numpy as np
import scipy


def perturb(pc: np.ndarray, magnitude: float = 0.01) -> np.ndarray:
    """
    Perturb each point in the input point cloud, with random 3D vectors
    
    Parameters:
        pc: np.ndarray
            xyz values of the point cloud, shaped like (n, 3)
        magnitude: float
            maximum magnitude of the randomly oriented vectors, added
            to the xyz values
    
    Returns:
        np.ndarray
            the perturbed xyz values
                
    Random points on the sphere are generated with a gaussian distribution
    as per https://mathworld.wolfram.com/SpherePointPicking.html
    """
    unit_vectors = np.random.randn(*pc.shape)
    unit_vectors /= np.linalg.norm(unit_vectors, axis=1)[:,np.newaxis]
    return pc + unit_vectors * np.random.randn(len(unit_vectors), 1) * magnitude


def rotate(pc: np.ndarray, rot_center: tuple[float] = (0, 0), flip: bool = True) -> np.ndarray:
    """
    Random uniform rotation around an axis, perpendicular to the XY plane
    and centered on rot_center (x, y)
    
    Parameters:
        pc: np.ndarray
            xyz values of the point cloud, shaped like (n, 3)
        rot_center: tuple[float]
            location of rotation axis, perpendicular to the XY plane, default (0,0)
        flip: bool
            randomly flip along X or Y axis, default True
    
    Returns:
        np.ndarray
            rotated xyz values
    
    https://www.euclideanspace.com/maths/geometry/affine/aroundPoint/matrix2d/
    """
    rot_mat = (
        scipy.stats.ortho_group.rvs(2) if flip
        else scipy.stats.special_ortho_group.rvs(2)
    )
    if rot_center == (0, 0):
        return np.hstack([
            np.dot(rot_mat, pc[:, :2].T).T,
            pc[:, 2, np.newaxis] 
        ])
    x, y = rot_center
    rot_mat = np.array([
        [rot_mat[0,0], rot_mat[0,1], x - rot_mat[0,0]*x - rot_mat[0,1]*y],
        [rot_mat[1,0], rot_mat[1,1], y - rot_mat[1,0]*x - rot_mat[1,1]*y],
        [0, 0, 1]
    ])
    xy_temp = np.hstack([pc[:, :2], np.ones((len(pc), 1))])
    xy = np.dot(rot_mat, xy_temp.T).T[:, :2]
    return np.hstack((xy, pc[:, 2, np.newaxis]))
