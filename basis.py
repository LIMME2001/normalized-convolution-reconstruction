"""Polynomial basis functions for local signal modeling."""

import numpy as np

class PolynomialBasis:
    """
    Creates polynomial basis functions up to a given order.
    
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
    
    @property
    def size(self):
        return self.B.shape[1]