"""Gaussian applicability function"""

import numpy as np

class GaussianApplicability:
    """Generates a 1D Gaussian kernel of a given size and sigma."""

    def __init__(self, window_size=7, sigma=2.0):
        self.size = window_size
        self.sigma = sigma
        self.kernel = self._build()

    def _build(self):
        x = np.arange(-(self.size//2), self.size//2 + 1)
        g = np.exp(-x**2 / (2 * self.sigma**2))
        return g/g.sum()

    def __call__(self):
        return self.kernel
