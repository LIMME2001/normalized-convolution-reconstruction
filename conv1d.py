import numpy as np
from scipy.signal import convolve
from basis import PolynomialBasis

class NormalizedConvolution1DUncertain:
    """Handles signals with missing data using a certainty vector."""

    def __init__(self, basis, applicability):
        self.B = basis()
        self.a = applicability()

    def filter_signal(self, signal):
        """Compute filter outputs in dual coordinates for the signal."""
        return np.array([convolve(signal, (b*self.a)[::-1], mode='same') for b in self.B])

    def compute_metric(self, certainty):
        """
        Compute the metric G for uncertain data.
        Requires a certainty matrix of the same shape as signal.
        
        Returns G:
            Metric matrix at each position according to: G_ij[n] = sumover_k(certainty[n-k]⋅a[k]⋅b_i[k]⋅b_j[k])
            Shape: (num_basis, num_basis, len(signal)).
        """
        num_basis = self.B.shape[0]
        N = len(certainty)
        G = np.zeros((num_basis, num_basis, N))

        for i in range(num_basis):
            for j in range(num_basis):
                f = (self.B[i]*self.a*self.B[j])[::-1]
                G[i, j, :] = convolve(certainty, f, mode='same')

        return G

    def compute_coordinates(self, signal, certainty):
        """
        Compute proper coordinates c taking uncertain data into account.

        Returns c:
            Coordinates for each basis function at each position,
            Shape: (num_basis, len(signal))
        """
        H = self.filter_signal(signal) # shape: (num_basis, N)
        G = self.compute_metric(certainty) # shape: (num_basis, num_basis, N)
        num_basis, N = H.shape
        c = np.zeros_like(H)

        # Solve linear system at each position
        for k in range(N):
            try:
                c[:, k] = np.linalg.solve(G[:, :, k], H[:, k])
            except np.linalg.LinAlgError:
                # Fallback if G is singular
                c[:, k] = np.zeros(num_basis)

        return c