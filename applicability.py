"""Gaussian applicability function"""

import numpy as np

class GaussianApplicability1D:
    """Generates a 1D Gaussian kernel of a given size and sigma."""

    def __init__(self, window_size=7, sigma=2.0):
        self.size = window_size
        self.sigma = sigma
        self.kernel = self._build()

    def _build(self):
        x = np.arange(-(self.size//2), self.size//2 + 1)
        a = np.exp(-x**2 / (2 * self.sigma**2))
        return a/a.sum()

    def __call__(self):
        return self.kernel

class GaussianApplicability2D:
    """Generates a 2D Gaussian applicability function."""

    def __init__(self, window_size=7, sigma=2.0):
        self.window_size = window_size
        self.sigma = sigma

    def __call__(self):
        r = self.window_size // 2
        x = np.arange(-r, r + 1)
        y = np.arange(-r, r + 1)
        X, Y = np.meshgrid(x, y, indexing="xy")

        return np.exp(-(X**2 + Y**2) / (2 * self.sigma**2))
