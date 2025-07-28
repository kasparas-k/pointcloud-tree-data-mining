from typing import Callable

import numpy as np
from scipy.optimize import curve_fit


def gauss(x: np.ndarray, a: float, x0: float, sigma: float) -> np.ndarray:
    x = np.array(x)
    return a * np.exp(-(x - x0)**2 / (2*sigma**2))


def fit_gaussian(feature: np.ndarray) -> tuple[float, float, float]:
    """
    Given an array of feature values, fit a gaussian curve to their histogram.

    Parameters:
        feature: np.ndarray
            Array of feature values to fit a gaussian curve to

    Returns:
        scale: float
            value at the gaussian's peak
        mean: float
            mean of the gaussian curve
        std: float
            standard deviation of the gaussian curve
    """
    y, x = np.histogram(feature, bins=20)
    x = np.array([np.mean(x[i:i+1]) for i in range(len(x) - 1)])

    freq_max = y.max()
    feat_mean = x.mean()
    feat_std = x.std()

    (scale, mean, std), _ = curve_fit(gauss, x, y, p0=[freq_max, feat_mean, feat_std])
    return scale, mean, std


def get_gauss_to_gauss_transform(
    x1: np.ndarray,
    x2: np.ndarray,
    right_margin: float,
    match_to_range: bool = False
) -> tuple[Callable[[np.ndarray], np.ndarray], tuple[float, float]]:
    """
    Given two arrays of features, get a transformation to transport x1 in such a way
    that when fitting a gaussian to it, the result would match a gaussian fit to x2

    Parameters:
        x1: np.ndarray
            Array containing the source distribution to transform
        x2: np.ndarray
            Array containing the target distribution to transform into
        right_margin: float
            Reduce the maximum allowable x argument for the returned transformation function
            (out of bound arguments get set to the value at the closest boundary)
        match_to_range: bool = False
            If set to True, match x1 to a gaussian the mean of which is the middle of
            the range of x2, and the standard deviation is 1/3 * range of x2.
            Essentially, this makes a distribution that covers the entire range of x2
            in a balanced way that does not take into the account the shape of x2

    Returns:
        Callable:
            function that linearly transforms points drawn from x1 to have a similar
            distribution to the gaussian fit of x2: f(x1) = scale*x1 + translation
        Transform parameters:
            (scale, translation) of the linear transformation
    """
    _, m1, s1 = fit_gaussian(x1)
    if not match_to_range:
        _, m2, s2 = fit_gaussian(x2)
    else:
        max2 = np.max(x2)
        min2 = np.min(x2)
        m2 = (max2 + min2) / 2
        s2 = (max2 - min2) / 6

    scale = s2 / s1
    translation = -scale * m1 + m2

    left = x2.min()
    right = x2.max() - right_margin
    def transform(x):
        x_t = scale*x + translation
        if isinstance(x, np.ndarray):
            x_t[x_t >= right] = right
            x_t[x_t <= left] = left
        else:
            if x_t >= right:
                x_t = right
            elif x_t <= left:
                x_t = left
        return x_t
    
    return transform, (scale, translation)
