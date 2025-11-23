"""
certainty.py

Certainty estimators for normalized convolution.
"""

import numpy as np
from scipy.ndimage import sobel, uniform_filter, binary_dilation

# Base class
class CertaintyBase:
    """Base class for certainty computation functions."""
    def compute(self, img):
        raise NotImplementedError("Certainty computation must be implemented in subclasses.")

# Edge-based certainty
class EdgeCertainty(CertaintyBase):
    """
    Certainty based on edge strength. 
    Strong edges = high certainty, flat regions = lower certainty
    """
    def __init__(self, min_cert=0.2):
        self.min_cert = min_cert

    def compute(self, img):
        gx = sobel(img, axis=1)
        gy = sobel(img, axis=0)
        mag = np.hypot(gx, gy)
        mag = mag/(mag.max() + 1e-8)

        C = self.min_cert + (1 - self.min_cert)*mag
        return np.clip(C, 0, 1)


# Noise-based certainty
class NoiseCertainty(CertaintyBase):
    """
    Certainty based on local variance.
    High noise = low certainty.
    """
    def __init__(self, window=5):
        self.window = window

    def compute(self, img):
        mean = uniform_filter(img, self.window)
        mean_sq = uniform_filter(img**2, self.window)
        var = mean_sq - mean**2

        var_norm = var / (var.max() + 1e-8)
        return np.clip(1 - var_norm, 0, 1)


# Segmentation-based certainty
class SegmentationCertainty(CertaintyBase):
    """
    Certainty based on segmentation boundaries.
    Pixels near boundaries are considered less certain.
    """
    def __init__(self, boundary_width=3, boundary_cert=0.3):
        self.boundary_width = boundary_width
        self.boundary_cert = boundary_cert

    def compute(self, segmentation):
        boundaries = np.zeros_like(segmentation, dtype=bool)

        # Detect boundaries by adjacent label mismatch
        boundaries[:-1, :] |= segmentation[:-1, :] != segmentation[1:, :]
        boundaries[:, :-1] |= segmentation[:, :-1] != segmentation[:, 1:]

        # Expand boundary region
        boundary_zone = binary_dilation(boundaries, iterations=self.boundary_width)

        C = np.ones_like(segmentation, dtype=float)
        C[boundary_zone] = self.boundary_cert
        return C


# Certainty combiner
def combine_certainties(*certainties):
    """
    Combine multiple certainty maps by multiplication.
    """
    C = np.ones_like(certainties[0], dtype=float)
    for c in certainties:
        C *= c
    return np.clip(C, 0, 1)