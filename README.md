# normalized-convolution-reconstruction

This project implements the theory from **Knutsson et al.** on **Normalized Convolution**, a principled method for computing **local signal approximations** even when the data contains unknown or unreliable samples. The goal is to provide a Python framework that can handle **1D signals and 2D images** with missing or uncertain data and reconstruct them accurately using polynomial basis functions.

## Features

- **1D Signal Reconstruction:**  
  - Handles missing samples using a certainty vector.  
  - Supports polynomial basis up to arbitrary order (Taylor-like expansions).  
  - Reconstructs both the signal and its local derivatives.  
  - Visualizes the original signal, signal with missing samples and reconstructed signal coefficients.

- **Polynomial Basis Functions:**  
  - Flexible order selection (linear, quadratic, cubic, etc.).  

- **Gaussian Applicability Function:**  
  - Generates 1D Gaussian kernels to weight local neighborhoods.  
  - Used for smoothing and creating the normalized convolution filters.

## Installation

```bash
git clone https://github.com/LIMME2001/normalized-convolution-reconstruction.git
cd normalized-convolution-reconstruction

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
<!-- 
#TODO: Implement certainty from edge detectors, noise estimators, segmentation-->

## Current Progress

- Implemented 1D normalized convolution with uncertain signals.
- Polynomial basis functions up to arbitrary order with factorial scaling.
- Gaussian applicability function implemented for 1D signals.
- Example scripts for testing and visualizing reconstruction with multiple basis orders.

## TODO

- Extend to 2D:
  - Implement 2D polynomial basis and Gaussian applicability.
  - Handle missing or uncertain pixels and reconstruct full images.

## References

- Knutsson, H., Westin, C.-F., & others. *Normalized Convolution: A Framework for Handling Uncertain Data in Signal Processing*.

