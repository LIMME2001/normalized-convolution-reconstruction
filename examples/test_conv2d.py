import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from imageio import imread

from applicability import GaussianApplicability2D
from basis import PolynomialBasis2D
from conv1d import NormalizedConvolution2D

# ================================================================================= #
# User-configurable section                                                         #
# ================================================================================= #
IMAGE_PATH = "Scalespace0.png"  # path to image, fallback to synthetic if not found #
GRID_SIZE = 7                   # window size for basis/applicability               #
POLY_ORDER = 0                  # order of polynomial basis                         #
SIGMA = np.sqrt(2)              # Gaussian sigma                                    #
CERTAINTY_PROB = 0.8            # probability that a pixel is missing               #
USE_SYNTHETIC = False           # force synthetic image instead of loading          #
# ================================================================================= #

# Load image or generate synthetic
try:
    if USE_SYNTHETIC:
        raise FileNotFoundError
    img = imread(IMAGE_PATH)
    if img.ndim == 3:
        img = img.mean(axis=2) # convert RGB to grayscale
    img = img.astype(float)
except Exception:
    print("Using synthetic test image.")
    x = np.linspace(0, 4*np.pi, 200)
    X, Y = np.meshgrid(x, x)
    img = np.sin(X) + 0.2*np.cos(Y*2)

# Normalize image to 0–1
img = img / img.max()

# Create certainty mask and masked image
certainty = (np.random.rand(*img.shape) > CERTAINTY_PROB).astype(float)
masked = img * certainty

# Build basis and applicability
basis = PolynomialBasis2D(order=POLY_ORDER, window_size=GRID_SIZE)
app = GaussianApplicability2D(window_size=GRID_SIZE, sigma=SIGMA)

# Normalized convolution
nc = NormalizedConvolution2D(basis, app)
c = nc.compute_coordinates(masked, certainty)
imlp = nc.low_pass_filtered(masked, certainty)
reconstructed = c[0] # constant basis coefficient as the main reconstruction

# Plot results
plt.figure(figsize=(12, 12))

plt.subplot(2, 2, 1)
plt.title("Original image")
plt.imshow(img, cmap="gray")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.title("Masked (uncertain) image")
plt.imshow(masked, cmap="gray")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.title("Low-pass filtered (imlp)")
plt.imshow(imlp, cmap="gray")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.title("Reconstructed (normalized convolution)")
plt.imshow(reconstructed, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()