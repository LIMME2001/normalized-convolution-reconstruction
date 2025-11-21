import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt

from basis import PolynomialBasis
from applicability import GaussianApplicability
from conv1d import NormalizedConvolution1D

# 1D test signal
k = np.arange(0, 200)
signal = np.sin(k / 15) + 0.1 * np.sin(k / 3) # more advanced signal

# Create basis + applicability
order = 3
basis = PolynomialBasis(order=order, window_size=9)
applicability = GaussianApplicability(window_size=9, sigma=2.0)

# Create NC operator
nc = NormalizedConvolution1D(basis, applicability)
c = nc.compute_coordinates(signal)

# Plot
num_basis = order + 1
plt.figure(figsize=(14, 2.5 * (num_basis + 1)))
plt.suptitle("Normalized Convolution - Full Certainty", fontsize=16)
plt.subplot(num_basis + 1, 1, 1)
plt.plot(signal, label="Original signal", linewidth=1.5)
plt.plot(c[0], "--", label="Reconstructed (c₀)", linewidth=1.2)
plt.legend()
plt.grid(True)

# Plot higher-order coefficients
for i in range(1, num_basis):
    plt.subplot(num_basis + 1, 1, i + 1)
    plt.plot(c[i], label=f"c{i} (Basis coefficient)", linewidth=1.2)
    plt.legend()
    plt.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
