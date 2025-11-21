import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from basis import PolynomialBasis
from applicability import GaussianApplicability
from conv1d import NormalizedConvolution1D

# 1D test signal
k = np.arange(0, 101)
signal = np.sin(k/10) + 0.1*np.sin(k/2) # Slightly more complex signal

# Uncertain signal
certainty = (np.random.rand(len(signal)) > 0.3).astype(float)
signal_uncertain = signal * certainty

# Adjustable polinomial order
orders_to_test = [1, 2, 3]

for order in orders_to_test:
    print(f"\n=== Testing basis order {order} ===")

    # Basis and applicability
    basis = PolynomialBasis(order=order, window_size=7)
    applicability = GaussianApplicability(window_size=7, sigma=2.0)

    # Normalized convolution
    nc = NormalizedConvolution1D(basis, applicability)
    c = nc.compute_coordinates(signal_uncertain, certainty) # Shape: (order+1, len(signal))

    # Create figure
    fig, axes = plt.subplots(order + 2, 1, figsize=(12, 2*(order+2)), sharex=True)
    fig.suptitle(f"Normalized Convolution - Polynomial order {order}", fontsize=16)

    # Top plot: original vs missing vs c0
    axes[0].plot(signal, '--', label='Original signal')
    axes[0].plot(signal_uncertain, '-', label='Signal with missing samples')
    axes[0].plot(c[0, :], '-', label='Recovered signal c0', linewidth=2)
    axes[0].set_ylabel("Amplitude")
    axes[0].legend()
    axes[0].grid(True)

    # Plot all coefficients
    for i in range(order + 1):
        axes[i+1].plot(c[i, :], label=f'c{i} (order {i})')
        axes[i+1].set_ylabel(f'c{i}')
        axes[i+1].legend()
        axes[i+1].grid(True)

    axes[-1].set_xlabel("Sample index")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
