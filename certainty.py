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
