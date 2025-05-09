#%%

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """
    Sigmoid function.
    
    Parameters:
    - x: ndarray or scalar - The input to the sigmoid function.
    
    Returns:
    - The result of applying the sigmoid function element-wise.
    """
    return 1 / (1 + np.exp(-x))

def sum_of_sigmoid(x, a, b, c, d):
    """
    Implements a sum of sigmoid function and generates synthetic data.
    
    Parameters:
    - x: ndarray of shape (n_samples, n_features) - The input data.
    - a: ndarray of shape (n_features, n_terms) - Coefficients for the features.
    - b: ndarray of shape (n_terms,) - Bias terms for the sigmoid functions.
    - c: ndarray of shape (n_terms,) - Coefficients for the sum of sigmoid terms.
    - d: ndarray of shape (1,) - An additional bias term to be added to the output.
    
    Returns:
    - output: ndarray of shape (n_samples, 1) - The output from the sum of sigmoid functions.
    """
    if x.ndim == 1:
        x = x.reshape(-1,1)
    n_features = x.shape[1]
    
    # Initialize a list to collect the results
    result = []
    
    # TODO: do the following operation in a vector manner instead of for loop
    for i in range(n_features):
        # Multiply each column of x (x[:, i]) with a (1, n_terms)
        term = c.T * sigmoid(np.dot(x[:, [i]], a) + b) + d  # Broadcasting handled here
        result.append(term[:,:,np.newaxis])  # Append the result
    
    # Concatenate the result list along the second axis (axis=1) to get shape (100, n_features)
    sigmoid_terms = np.concatenate(result, axis=-1)
    output = np.sum(sigmoid_terms, axis=1)
    
    # If the input x is 1D, we need to squeeze the output to match the shape (n_samples,)
    if x.ndim == 1:
        output = output.squeeze()

    if n_features == 1:
        output = output.squeeze()

    return output

# Function to generate synthetic data
def generate_synthetic_data(n_samples, n_features, n_terms, seed=None, min_x=-2, max_x=2):
    """
    Generates synthetic data to be used in the sum of sigmoid function.
    
    Parameters:
    - n_samples: int - Number of data points.
    - n_features: int - Number of features in the data.
    - n_terms: int - Number of sigmoid terms.
    
    Returns:
    - x: ndarray of shape (n_samples, n_features) - The input data.
    - a: ndarray of shape (n_features, n_terms) - Coefficients for the features.
    - b: ndarray of shape (n_terms,) - Bias terms for the sigmoid functions.
    - c: ndarray of shape (n_terms,) - Coefficients for the sum of sigmoid terms.
    - d: ndarray of shape (1,) - An additional bias term.
    """
    np.random.seed(seed) if seed else None
    # Randomly generate input data (x)
    x = np.random.rand(n_samples, n_features)*(max_x-min_x)+min_x
    # x = np.linspace(-5, 5, n_samples, n_features)
    
    # Randomly generate the coefficients and bias terms
    a = np.random.randint(0, 50, size=(n_features, n_terms))
    b = np.random.randint(-50, 50, size=(n_terms))
    c = np.random.randint(0, 50, size=(n_terms))
    d = np.random.randint(-50, 50, size=1)
    
    return x, a, b, c, d

# Example usage
n_samples = 1000  # Number of samples
n_features = 1   # Number of features
n_terms = 5      # Number of sigmoid terms

seed = 1
# Generate synthetic data
x, a, b, c, d = generate_synthetic_data(n_samples, n_features, n_terms, seed=seed, min_x=0, max_x=25)

# Compute the output using the sum of sigmoid function
output = sum_of_sigmoid(x, a, b, c, d)

# print("Synthetic Data Output:\n", output)

idx = np.argsort(x, axis=0)
plt.plot(x[idx,0], output[idx])
plt.show()



#%% manually set parametsr
# multidimensional input
# x = np.linspace(-2,2,100).reshape(-1,1)
# x = np.linspace(-2,2,100).reshape(-1,1)
x = np.random.rand(100,3)*4-2
# x[:,0] = x[:,0]*4-2
# x[:,1] = x[:,1]*3
a = np.array([[20, 48, 22, 37, 13]])
b = np.array([ 47,   3,  34, -40,  46])
c = np.array([25, 21, 39, 32, 19])
d = np.array([-25])

output = sum_of_sigmoid(x, a, b, c, d)
idx = np.argsort(x[:,0])
plt.plot(x[idx,0], output[idx,0])
idx = np.argsort(x[:,1])
plt.plot(x[idx,1], output[idx,1])
plt.show()

#%% manually set parameters
x = np.linspace(-2,2,100)
a = np.array([0, 5, 1, 2])
b = np.array([2, 10, 20, -1])
c = np.array([1, 4, 2, 5])
d = np.array([0, 0, 0, 1])
output = sum_of_sigmoid(x[:,np.newaxis], a[np.newaxis,:], b, c, d)

# print("Synthetic Data Output:\n", output)

plt.plot(x, output)
plt.show()