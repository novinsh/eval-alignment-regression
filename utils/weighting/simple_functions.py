#%%
# Weighting and Chaining functions implemented here are based on the following paper:
# Allen, Sam, David Ginsbourger, and Johanna Ziegel. "Evaluating forecasts for high-impact events using transformed kernel scores." SIAM/ASA Journal on Uncertainty Quantification 11.3 (2023): 906-940.
# Visualization based on Fig. 1 of their paper.

# These weighting functions are used in our controlled experiments as a simple case 
# for our alignment pipeline. We generated synthetic downstream scores given a known
# weighting/chaining function from the ones implemented here and expect the alignment
# model to learn them by minimizing the distance between the upstream and downstream
# objective functions.

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Define weight functions
def w1(x, t):
    # return (x > t).astype(float)
    return np.heaviside(x-t, 0)

def w2(x, a, b):
    # return ((x > a) & (x < b)).astype(float)
    return np.heaviside(x - a,0) * np.heaviside(b-x, 0)

def w3(x, mu, sigma):
    return norm.cdf(x, mu, sigma)

# Define chaining functions
def v1(x, t):
    return np.maximum(x, t)

def v2(x, a, b):
    return np.minimum(np.maximum(x, a), b)

def v3(x, t, mu, sigma):
    return (x - t) * norm.cdf(x, mu, sigma) + sigma**2 * norm.pdf(x, mu, sigma)

# Define kernels
def kernel(v_func, x, x_prime, *args):
    return np.abs(v_func(x, *args) - v_func(x_prime, *args))

# Generate data for the weight functions
t = 0.5
x = np.linspace(-2, 2, 500)
weights = [w1(x, t), w2(x, -0.5, 1.5), w3(x, 0, 1)]

# Plot weight functions
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
weight_titles = ["w(z) = 1{z > t}", "w(z) = 1{a < z < b}", "w(z) = \u03A6(z; \u03BC, \u03C3)"]

for ax, weight, title in zip(axes, weights, weight_titles):
    ax.plot(x, weight, color='blue')
    ax.set_title(title)
    ax.set_xlabel("z")
    ax.set_ylabel("w(z)")

plt.tight_layout()
plt.show()

# Generate data for the chaining functions
x = np.linspace(-2, 2, 500)
chaining = [v1(x, t), v2(x, -0.5, 1.5), v3(x, t, 0, 1)]

# Plot chaining functions
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
chaining_titles = ["v(z) = max(z, t)", "v(z) = min(max(z, a), b)", "v(z) = (z-t)\u03A6(z) + \u03C3^2\u03C6(z)"]

for ax, chain, title in zip(axes, chaining, chaining_titles):
    ax.plot(x, chain, color='green')
    ax.set_title(title)
    ax.set_xlabel("z")
    ax.set_ylabel("v(z)")
plt.tight_layout()
plt.show()

# Generate a grid of x and x'
x = np.linspace(-2, 2, 100)
x_prime = np.linspace(-2, 2, 100)
X, X_prime = np.meshgrid(x, x_prime)

# Calculate kernels for each chaining function
kernel1 = kernel(v1, X, X_prime, t)
kernel2 = kernel(v2, X, X_prime, -0.5, 1.5)
kernel3 = kernel(v3, X, X_prime, t, 0, 1)

# Plot the kernels
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
kernels = [kernel1, kernel2, kernel3]
kernel_titles = ["k(z, z') = |v1(z) - v1(z')|", "k(z, z') = |v2(z) - v2(z')|", "k(z, z') = |v3(z) - v3(z')|"]

for ax, kernel, title in zip(axes, kernels, kernel_titles):
    im = ax.imshow(kernel, extent=[-2, 2, -2, 2], origin='lower', cmap='Reds')
    ax.set_title(title)
    ax.set_xlabel("z")
    ax.set_ylabel("z'")
    fig.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()