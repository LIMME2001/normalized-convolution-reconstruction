"""Polynomial basis functions for local signal modeling."""

import numpy as np

class PolynomialBasis1D:
    """
    Creates a 1D polynomial basis functions up to a given order.
    
    basis order k means [1, x, x^2, ..., x^k]
    """

    def __init__(self, order=2, window_size=7):
        self.order = order
        self.size = window_size
        self.x = np.arange(-(window_size//2), window_size//2 + 1)
        self.basis = self._build()

    def _build(self):
        # Build basis matrix (k+1 rows, window size columns)
        return np.vstack([self.x**n for n in range(self.order + 1)])
    
    def __call__(self):
        return self.basis
    
class PolynomialBasis2D:
    """
    2D polynomial basis up to a given total order.
    Produces basis functions on a (window_size x window_size) grid.
    """

    def __init__(self, order, window_size):
        self.order = order
        self.window_size = window_size

    def __call__(self):
        r = self.window_size // 2
        x = np.arange(-r, r + 1)
        y = np.arange(-r, r + 1)
        X, Y = np.meshgrid(x, y, indexing="xy")

        basis = []

        for i in range(self.order + 1):
            for j in range(self.order + 1 - i):
                # factorial scaling (consistent with 1D version)
                basis.append(X**i * Y**j)

        return np.array(basis)