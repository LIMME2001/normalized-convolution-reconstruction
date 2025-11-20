import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from basis import PolynomialBasis
from applicability import GaussianApplicability
from conv1d import NormalizedConvolution1DUncertain

# 1D test signal
k = np.arange(0, 101)
signal = np.sin(k/10)

# Basis and applicability
basis = PolynomialBasis(order=2, window_size=7)
applicability = GaussianApplicability(window_size=7, sigma=2.0)

# Uncertain signal (change uncertainty if needed)
certainty = (np.random.rand(len(signal)) > 0.3)
signal_uncertain = signal * certainty

# Normalized convolution
nc = NormalizedConvolution1DUncertain(basis, applicability)
c = nc.compute_coordinates(signal_uncertain, certainty)

# Plot
plt.figure(figsize=(12,6))
plt.subplot(2,1,1)
plt.plot(signal_uncertain, label='Input signal with missing samples')
plt.plot(c[0,:], label='Recovered signal c0')
plt.legend()
plt.subplot(2,1,2)
plt.plot(c[1,:], label='Recovered derivative c1')
plt.legend()
plt.show()
