# This file contains implementation simple regression models with syntehtic data 
# synthetic data is generated using either of homoskedastic or heteroskedastic noise
#%%

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generate synthetic dataset

def generate_data(data_type=1, noise_type=1, n_observations = 1000):
    # noise_type: 1 homoskedastic, 2 heteroskedastic
    if data_type==1:
        X_ = np.random.uniform(-10, 10, size=(n_observations, 1))
        if noise_type == 1:
            noise = np.random.normal(0, 1.0, (n_observations, 1)) 
        else:
            noise = np.random.normal(0, 1.0, (n_observations, 1)) * np.abs(X_)
        y_ = 0.5 * X_ + np.sin(X_) + 0.25 * noise  # Some non-linear function with noise
    elif data_type==2:
        X_ = np.random.uniform(0, 5, size=(n_observations, 1))
        # Heteroskedastic noise: variance increases with the magnitude of X
        if noise_type == 1:
            noise = np.random.normal(0,1, size=(n_observations, 1))
        else:
            noise = np.random.randn(n_observations, 1) * (0.5 * np.abs(X_)**2)  # Larger variance for larger X
        y_ = 2 * X_**2 + noise  # Quadratic relationship with heteroskedastic noise
    else:
        assert False, "datatype does not exist"
    return X_, y_

#
data_type = 1
noise_type = 2
n_observations = 1000
X_, y_ = generate_data(data_type=data_type, noise_type=noise_type, n_observations=n_observations)
X = torch.tensor(X_, dtype=torch.float32)
y = torch.tensor(y_, dtype=torch.float32)

plt.title(f"synthetic data with {'homoskedastic' if noise_type == 1 else 'heteroskedastic'} noise")
plt.plot(X, y, '.')
plt.xlabel('x')
plt.ylabel('y')
# plt.savefig('../figs/synth_reg/')
plt.show()
#%% Quantile Model with quantile loss

# Define a simple neural network for quantile regression
class QuantileRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QuantileRegressionModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.output_layer(x)

# Define the quantile loss
def quantile_loss(y_true, y_pred, tau):
    error = y_true - y_pred.unsqueeze(1)
    loss = torch.mean(torch.max((tau - 1) * error, tau * error))
    return loss

# Initialize the model and optimizer
n_quantiles = 10
quantiles = torch.linspace(0.05, 0.95, n_quantiles)  # 100 quantiles from 0 to 1, equidistant
model = QuantileRegressionModel(input_dim=1, output_dim=n_quantiles)  # Output for each quantile
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Make predictions (outputs 100 values for each sample)
    y_pred = model(X)

    # Calculate quantile loss for all quantiles
    losses = []
    for tau in quantiles:
        loss = quantile_loss(y, y_pred[:, int(tau * n_quantiles)], tau)  # Get the prediction for the specific quantile
        losses.append(loss)

    total_loss = torch.mean(torch.stack(losses))  # Average loss over all quantiles
    total_loss.backward()
    optimizer.step()

    # Print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item():.4f}')

# Plot the true function and predictions for some quantiles
model.eval()
y_pred = model(X).detach().numpy()

# Plot predictions for the first few quantiles
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='True Data', alpha=0.5)

idx = np.argsort(X[:,0], axis=0)

# Plot the predictions for a few quantiles
for i, tau in enumerate([0.01, 0.25, 0.5, 0.75, 0.99]):
    plt.plot(X[idx,0], y_pred[idx, int(tau * n_quantiles)], label=f'Quantile {tau:.2f}')

plt.title("Quantile Model Predictions vs True Data")
plt.legend()
plt.show()


# Plot the true function and prediction intervals
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='True Data', alpha=0.5)

for q in range(n_quantiles//2-1):
    lower_quantile = quantiles[q]
    upper_quantile = quantiles[-q-1]
    lower_bound = y_pred[idx, int(lower_quantile * n_quantiles)]
    upper_bound = y_pred[idx, int(upper_quantile * n_quantiles)]

    # Plot the prediction interval (5th to 95th quantile)
    pinterval = upper_quantile-lower_quantile
    plt.fill_between(X[idx,0], lower_bound, upper_bound, color='red', alpha=(q+1)/n_quantiles, label=f'{pinterval*100:2.0f}% Prediction Interval')

# Plot the median prediction (50th quantile)
median_pred = y_pred[idx, int(0.5 * n_quantiles)]
plt.plot(X[idx], median_pred, color='red', linewidth=2, label='Median Prediction (50th Quantile)')

plt.title("Prediction Intervals based on the Quantiles Predictions")
plt.legend()
plt.show()

#%% Sampled-based Model with CRPS loss

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split 

# torch.manual_seed(42)

# Define the neural network architecture
class DistributionNetwork(nn.Module):
    def __init__(self, input_dim, n_latents):
        super(DistributionNetwork, self).__init__()
        
        # Fully connected layers to transform input to latent space
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, n_latents)
        
        # Mixture model parameters: 
        # n_samples -> number of components in the mixture model
        self.fc_mean = nn.Linear(n_latents, 1)
        self.fc_logvar = nn.Linear(n_latents, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        # Generate mixture parameters
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        
        return mean, logvar

    def sample(self, mean, logvar, n_samples=1):
        # Reparameterization trick for Gaussian sampling
        std = torch.exp(0.5 * logvar)
        eps = torch.randn((logvar.shape[0], n_samples))
        # eps = torch.randn(n_samples)
        # eps = torch.randn_like(std)
        return mean + eps * std

# CRPS loss function (using the provided implementation)
def crps_ensemble_torch(measurements, ensemble_predictions, v=None):
    pred_dim0 = ensemble_predictions.shape[0]
    pred_dim1 = ensemble_predictions.shape[1]
    
    def identity(h):
        return h
    v = identity if v is None else v
    
    obs_t = v(measurements.view(-1, 1)).view(pred_dim0)
    preds_t = v(ensemble_predictions.view(-1, 1)).view(pred_dim0, pred_dim1)
    
    # obs_diff = torch.abs(obs_t[:, torch.newaxis] - preds_t)
    obs_diff = torch.abs(torch.unsqueeze(obs_t, -1) - preds_t)
    pred_diff = torch.abs(torch.unsqueeze(preds_t, -1) - torch.unsqueeze(preds_t, -2))
    crps = obs_diff - 0.5 * pred_diff.mean(axis=(-2))
    
    return crps

# Training loop for the network
def train_model(model, data_loader, num_epochs=100):
    optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    
    for epoch in range(num_epochs):
        model.train()
        
        total_loss = 0
        for inputs, targets in data_loader:
            optimizer.zero_grad()

            # Get distribution parameters
            mean, logvar = model(inputs)

            # print(mean.shape)
            # print(inputs.shape)
            
            # Generate samples from the distribution
            samples = model.sample(mean, logvar, n_samples=100)
            # print(samples.shape)
            # assert False
            
            # Calculate CRPS loss
            crps_loss = crps_ensemble_torch(targets, samples)
            loss = crps_loss.mean()
            
            # Backpropagate and optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(data_loader)}')

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

np.random.seed(42)

# Example usage
input_dim = 1  # Input feature size
n_latents = 32  # Latent space size
model = DistributionNetwork(input_dim, n_latents)

# Assuming you have a data loader with inputs and targets
dataset = CustomDataset(X, y)

# Calculate split lengths
train_size = int(0.7 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size

# Split dataset into train, validation, and test sets
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Example: Check the first batch of each DataLoader
for inputs, targets in train_loader:
    print(f"Train batch: {inputs.shape}, {targets.shape}")
    break

for inputs, targets in val_loader:
    print(f"Validation batch: {inputs.shape}, {targets.shape}")
    break

for inputs, targets in test_loader:
    print(f"Test batch: {inputs.shape}, {targets.shape}")
    break

# Train the model
train_model(model, train_loader, num_epochs=100)


#%% Make prediction on the whole set - non batch
def dataset_to_nobatch(dataset):
    inputs = []
    targets = []
    for input_data, target_data in dataset:
        inputs.append(input_data)  
        targets.append(target_data)  
    inputs = torch.vstack(inputs)  
    targets = torch.vstack(targets)  
    return inputs, targets

X_tr, y_tr = dataset_to_nobatch(train_dataset)
X_val, y_val = dataset_to_nobatch(val_dataset)
X_te, y_te = dataset_to_nobatch(test_dataset)

# Inference - generating samples using the trained model
def generate_samples(model, x_input, n_samples=100, mean_bias=0, logvar_bias=0, return_mean=False):
    model.eval()
    with torch.no_grad():
        mean, logvar = model(x_input)
        # print(mean.shape)
        # print(logvar.shape)
        generated_samples = model.sample(mean+mean_bias, logvar+logvar_bias, n_samples=n_samples)
    if return_mean:
        return generated_samples, mean
    else:
        return generated_samples

y_tr_pred, y_tr_pred_mean = generate_samples(model, X_tr, return_mean=True)
y_val_pred, y_val_pred_mean = generate_samples(model, X_val, return_mean=True)
y_te_pred, y_te_pred_mean = generate_samples(model, X_te, return_mean=True)

def plot_prediction(X, y, y_pred, y_pred_mean=None, title='', save_filename=''):
    plt.title(title) if save_filename == '' else None
    idx = np.argsort(X[:,0])
    plt.plot(X[idx,0], y_pred[idx,:], '.', color='red', alpha=0.1,)
    plt.plot(X[idx,0], y[idx,0], '.', color='blue', label='observation $y$')
    plt.plot([], [], '.', color='red', alpha=0.8, label='prediction $\mathbf{\hat{y}}$')
    if y_pred_mean is not None: # use the mean directly provided by the model
        plt.plot(X[idx,0], y_pred_mean[idx,:], '-', color='black', label='mean prediction $\mathbb{E}[\mathbf{\hat{y}}]$')
    else: # calculate sample mean
        plt.plot(X[idx,0], y_pred[idx,:].mean(axis=1), '-', color='black', label='mean prediction $\mathbb{E}[\mathbf{\hat{y}}]$')
    plt.grid(alpha=0.25)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    if save_filename:
        plt.savefig(f"../figs/synth_reg/{save_filename}.pdf", format="pdf", bbox_inches = 'tight')

# whole dataset
idx = np.argsort(X[:,0])
y_pred, y_pred_mean = generate_samples(model, X, return_mean=True)
plot_prediction(X, y, y_pred, y_pred_mean, 'whole dataset')
plt.show()

# train
plt.figure(figsize=(12,3))
plt.subplot(1,3,1)
plt.title("train set")
plot_prediction(X_tr, y_tr, y_tr_pred, y_tr_pred_mean, 'train set')
# validation
plt.subplot(1,3,2)
plot_prediction(X_val, y_val, y_val_pred, y_val_pred_mean, 'validation set')
# test
plt.subplot(1,3,3)
plot_prediction(X_te, y_te, y_te_pred, y_te_pred_mean, 'test set')
plt.show()

# save figure for paper on test set:
data_type_str = "sinusoidal" if data_type == 1 else "quadratic"
noise_type_str = "homo" if noise_type == 1 else "hetero"
plot_prediction(X_te, y_te, y_te_pred, y_te_pred_mean, save_filename=f'result_data_{data_type_str}-noise_{noise_type_str}')
plt.show()

#%% save as a dataset for alignment

def save_dataset(X, y, y_pred, suffix=''):
    arr = np.concatenate([X.numpy(),y.numpy(),y_pred.numpy()],axis=1)
    np.save(f"datasets/data{data_type}{noise_type}_model_{suffix}.npy", arr)

# optimal prediction
# whole dataset
y_pred = generate_samples(model, X, mean_bias=0, logvar_bias=0)
plot_prediction(X, y, y_pred, title='whole dataset (optimal model)')
plt.show()
# validation and test
y_val_pred = generate_samples(model, X_val, mean_bias=0, logvar_bias=0)
y_te_pred = generate_samples(model, X_te, mean_bias=0, logvar_bias=0)
save_dataset(X, y, y_pred, suffix='optimal')
save_dataset(X_val, y_val, y_val_pred, suffix='optimal_val')
save_dataset(X_te, y_te, y_te_pred, suffix='optimal_test')

# mean biased
mean_bias = np.random.normal(1,0.5,len(X)).reshape(-1,1)
y_pred = generate_samples(model, X, mean_bias=mean_bias, logvar_bias=0)
plot_prediction(X, y, y_pred, title='whole dataset (biased model)')
plt.show()
# arr = np.concatenate([X.numpy(),y.numpy(),y_pred.numpy()],axis=1)
# np.save(f"synthetic_regression_data/data{data_type}{noise_type}_model_biased.npy", arr)
# validation and test
y_val_pred = generate_samples(model, X_val, mean_bias=mean_bias[train_size:train_size+val_size], logvar_bias=0)
y_te_pred = generate_samples(model, X_te, mean_bias=mean_bias[-test_size:], logvar_bias=0)
save_dataset(X, y, y_pred, suffix='biased')
save_dataset(X_val, y_val, y_val_pred, suffix='biased_val')
save_dataset(X_te, y_te, y_te_pred, suffix='biased_test')

# high variance
logvar_bias = np.random.normal(0,1,len(X)).reshape(-1,1)
# logvar_bias = 1
y_pred = generate_samples(model, X, mean_bias=0, logvar_bias=logvar_bias)
plot_prediction(X, y, y_pred, title='whole dataset (high variance model)')
plt.show()
# arr = np.concatenate([X.numpy(),y.numpy(),y_pred.numpy()],axis=1)
# np.save(f"synthetic_regression_data/data{data_type}{noise_type}_model_highvar.npy", arr)
y_val_pred = generate_samples(model, X_val, mean_bias=0, logvar_bias=logvar_bias[train_size:train_size+val_size])
y_te_pred = generate_samples(model, X_te, mean_bias=0, logvar_bias=logvar_bias[-test_size:])
save_dataset(X, y, y_pred, suffix='highvar')
save_dataset(X_val, y_val, y_val_pred, suffix='highvar_val')
save_dataset(X_te, y_te, y_te_pred, suffix='highvar_test')

# low variance
logvar_bias = np.random.normal(-1,0.1,len(X)).reshape(-1,1)
# logvar_bias = -1
y_pred = generate_samples(model, X, mean_bias=0, logvar_bias=logvar_bias)
plot_prediction(X, y, y_pred, title='whole dataset (low variance model)')
plt.show()
# arr = np.concatenate([X.numpy(),y.numpy(),y_pred.numpy()],axis=1)
# np.save(f"synthetic_regression_data/data{data_type}{noise_type}_model_lowvar.npy", arr)
y_val_pred = generate_samples(model, X_val, mean_bias=0, logvar_bias=logvar_bias[train_size:train_size+val_size])
y_te_pred = generate_samples(model, X_te, mean_bias=0, logvar_bias=logvar_bias[-test_size:])
save_dataset(X, y, y_pred, suffix='lowvar')
save_dataset(X_val, y_val, y_val_pred, suffix='lowvar_val')
save_dataset(X_te, y_te, y_te_pred, suffix='lowvar_test')

# X_tmp = arr[:,[0]]
# y_tmp = arr[:,[1]]
# y_pred_tmp = arr[:,2:]
# print(X_tmp.shape)
# print(y_tmp.shape)
# print(y_pred_tmp.shape)
# arr2 = np.load("data2_model1.npy")
# print(arr2.shape)

#%% Make predictions in batches

def generate_samples(model, data_loader, n_samples=100):
    model.eval()
    all_inputs = []
    all_targets = []
    all_samples = []
    with torch.no_grad():
        for inputs, targets in data_loader:
            mean, logvar = model(inputs)
            generated_samples = model.sample(mean, logvar, n_samples=n_samples)
            all_samples.append(generated_samples)
            all_inputs.append(inputs)
            all_targets.append(targets)
    all_inputs = torch.cat(all_inputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_samples = torch.cat(all_samples, dim=0)
    return all_inputs.detach().numpy(), all_targets.detach().numpy(), all_samples.detach().numpy()

X_tr, y_tr, y_tr_pred = generate_samples(model, train_loader)
X_val, y_val, y_val_pred = generate_samples(model, val_loader)
X_te, y_te, y_te_pred = generate_samples(model, test_loader)

def plot_prediction(X, y, y_pred, title=''):
    plt.title(title)
    plt.plot(X, y, '.')
    plt.plot(X, y_pred, '.', color='red', alpha=0.1)

plt.figure(figsize=(12,3))
plt.subplot(1,3,1)
plot_prediction(X_tr, y_tr, y_tr_pred, 'train set')
plt.subplot(1,3,2)
plot_prediction(X_val, y_val, y_val_pred, 'validation set')
plt.subplot(1,3,3)
plot_prediction(X_te, y_te, y_te_pred, 'test set')
plt.show()

#%% Linear Regression with MSE loss

class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 32)  # One in and one out
        self.hidden = torch.nn.Linear(32, 32)  # One in and one out
        self.output = torch.nn.Linear(32, 1)  # One in and one out
 
    def forward(self, x):
        x = torch.relu(self.linear(x))
        x = torch.relu(self.hidden(x))
        y_pred = self.output(x)
        # y_pred = self.linear(x)
        return y_pred
 

def train_model(model, data_loader, num_epochs=100):
    optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    criterion = torch.nn.MSELoss()
    for epoch in range(num_epochs):
        model.train()
    
        total_loss = 0
        for inputs, targets in data_loader:
            # Forward pass: Compute predicted y by passing 
            # x to the model
            preds = model(inputs)
        
            # Compute and print loss
            loss = criterion(preds, targets)
        
            # Zero gradients, perform a backward pass, 
            # and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(data_loader)}')


linear_model = LinearRegressionModel()
train_model(linear_model, train_loader)

#%%
# whole dataset
y_pred = linear_model(X).detach().numpy()

plt.plot(X, y, '.')
plt.plot(X, y_pred, '.')
plt.show()

irreduceable_noise = ((np.mean(y_pred) - y)**2)#.mean()
variance = ((np.mean(y_pred) - y_pred)**2)#.mean()
# plt.plot(np.sort(irreduceable_noise, axis=0))
# plt.plot(np.sort(variance, axis=0))
# plt.show()
print(variance.mean())
print(irreduceable_noise.mean())

y_pred_2 = np.random.normal(y_pred, np.sqrt(irreduceable_noise.mean()))

plt.plot(X, y, '.')
plt.plot(X, y_pred_2, '.')
plt.show()

# test set
y_te_pred = linear_model(X_te)
plt.plot(X_te, y_te.detach().numpy(), '.')
plt.plot(X_te, y_te_pred.detach().numpy(), '.')
plt.show()
# %%
