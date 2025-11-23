import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from imageio import imread

from basis import PolynomialBasis2D
from applicability import GaussianApplicability2D
from conv1d import NormalizedConvolution2D
from certainty import EdgeCertainty, NoiseCertainty, SegmentationCertainty, combine_certainties


# ================================================================================= #
# User-configurable section                                                         #
# ================================================================================= #
IMAGE_PATH = "Scalespace0.png"
WINDOW_SIZE = 7
POLY_ORDER = 1
SIGMA = np.sqrt(2)

USE_SYNTHETIC_IMAGE = False
# ================================================================================= #

# Load image
try:
    if USE_SYNTHETIC_IMAGE:
        raise FileNotFoundError

    img = imread(IMAGE_PATH)
    if img.ndim == 3:
        img = img.mean(axis=2)
    img = img.astype(float)

except Exception:
    print("Using synthetic test image.")
    x = np.linspace(0, 4*np.pi, 200)
    X, Y = np.meshgrid(x, x)
    img = np.sin(X) + 0.2*np.cos(2 * Y)

img = img / img.max()

# Create certainty maps
Cedge = EdgeCertainty(min_cert=0.3).compute(img)
Cnoise = NoiseCertainty(window=5).compute(img)

# Make a fake segmentation
seg = (img > 0.5).astype(int)
Cseg = SegmentationCertainty(boundary_width=3, boundary_cert=0.4).compute(seg)

# Combine all certainties
certainty = combine_certainties(Cedge, Cnoise, Cseg)

# Mask the image using certainty (simulate missing data)
masked = img * certainty

# Run normalized convolution
basis = PolynomialBasis2D(order=POLY_ORDER, window_size=WINDOW_SIZE)
app = GaussianApplicability2D(window_size=WINDOW_SIZE, sigma=SIGMA)

nc = NormalizedConvolution2D(basis, app)
coords = nc.compute_coordinates(masked, certainty)
reconstructed = coords[0]   # constant coefficient = reconstructed image

# Plot
plt.figure(figsize=(14, 12))

plt.subplot(2, 3, 1)
plt.title("Original image")
plt.imshow(img, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.title("Combined Certainty Map")
plt.imshow(certainty, cmap="inferno")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.title("Masked image")
plt.imshow(masked, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.title("Reconstructed image")
plt.imshow(reconstructed, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.title("Edge Certainty")
plt.imshow(Cedge, cmap="inferno")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.title("Noise Certainty")
plt.imshow(Cnoise, cmap="inferno")
plt.axis("off")

plt.tight_layout()
plt.show()
