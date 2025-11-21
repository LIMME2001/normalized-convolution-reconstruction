import numpy as np
from scipy.signal import convolve, convolve2d

class NormalizedConvolution1D:
    """
    Normalized convolution for 1D signals.
    Certainty is optional. If omitted, all samples are treated as fully certain.
    """

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

    def compute_coordinates(self, signal, certainty=None):
        """
        Compute proper coordinates c taking uncertain data into account.

        Returns c:
            Coordinates for each basis function at each position,
            Shape: (num_basis, len(signal))
        """
        # Certainty optional
        if certainty is None:
            certainty = np.ones_like(signal, dtype=float)
        
        # Ensure correct shape
        certainty = np.asarray(certainty)
        if certainty.shape != signal.shape:
            raise ValueError("certainty must have same shape as signal")

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
    

class NormalizedConvolution2D:
    """
    2D Normalized Convolution using polynomial basis B(x,y) and applicability a(x,y).
    Certainty is optional. If omitted, all samples are treated as fully certain.
    """

    def __init__(self, basis, applicability):
        self.B = basis() # shape: (num_basis, w, w)
        self.a = applicability() # shape: (w, w)
        self.num_basis = self.B.shape[0]

    def filter_signal(self, image, certainty):
        """
        Compute H_i = conv2((image * certainty), (b_i * a))
        """
        imc = image * certainty
        return np.array([
            convolve2d(imc, (b * self.a)[::-1, ::-1], mode='same')
            for b in self.B
        ])

    def compute_metric(self, certainty):
        """
        Compute metric tensor:
        G_ij = conv2(certainty, B_i * a * B_j)
        """
        h, w = certainty.shape
        G = np.zeros((self.num_basis, self.num_basis, h, w))

        for i in range(self.num_basis):
            for j in range(self.num_basis):
                kernel = (self.B[i] * self.a * self.B[j])[::-1, ::-1]
                G[i, j] = convolve2d(certainty, kernel, mode='same')

        return G

    def compute_coordinates(self, image, certainty=None):
        """
        Compute normalized convolution coordinates per pixel.
        """
        if certainty is None:
            certainty = np.ones_like(image)

        H = self.filter_signal(image, certainty)
        G = self.compute_metric(certainty)

        h, w = image.shape
        c = np.zeros((self.num_basis, h, w))

        for y in range(h):
            for x in range(w):
                try:
                    c[:, y, x] = np.linalg.solve(G[:, :, y, x], H[:, y, x])
                except np.linalg.LinAlgError:
                    c[:, y, x] = 0

        return c
    
    def low_pass_filtered(self, image, certainty=None):
        if certainty is None:
            certainty = np.ones_like(image)
        return convolve2d(image * certainty, self.a, mode='same')