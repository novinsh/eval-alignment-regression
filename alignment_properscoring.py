#%%
# This code can be run based on the results saved from either of the following experiments:
# 1. synthetic_data_and_probabilistic_regression.py: where a synthetic regression data 
#    is generated and a regression model is fitted. The weight function is known in this experiment.
# 2. optimization_test.py: where a downstream task is formulated and downstream scores 
#    are obtained from a profit function under two possible modes, 
#    a) predictivie distribution matching the test distribution, and 
#    b) the two distributions are not matching
#    The weight function is not explicitely defined but the analytical results from
#    analyzing the downstream function provides us with an insights as to what the 
#    weight function should be. The downstream setup is still synthetic in nature!
#    i.e. the prices/costs, and demand predictions are not based on real-data.

# TODO: the train/validation/test splits are not implemented yet!
# TODO: the implementation is based on unbounded values. It can be extended for 
#       bounded cases as well!

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import os 

# import neural components used in the alignment model
from utils.constrainted_monotonic_nn import MonotonicLinear
from utils.constrainted_monotonic_nn import normalize
from utils.constrainted_monotonic_nn import inverse_normalize

# import weighting/chaining functions 
from utils.weighting.sum_of_sigmoids_functions import sum_of_sigmoid
from utils.weighting.simple_functions import v1, v2, v3
from utils.weighting.beta_functions import beta_calib_func

import properscoring as ps
import scoringrules as sr

from scipy.stats import kendalltau
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

class MonotonicNN(nn.Module):
    def __init__(self):
        super(MonotonicNN, self).__init__()
        # for real downstream 
        self.layer1 = MonotonicLinear(1, 50, gate_func='relu', indicator=torch.tensor([1]),)
        self.layer2 = MonotonicLinear(50, 50, gate_func='relu', indicator=torch.tensor([1]*50),)
        self.layer4 = nn.Linear(50,1) # use only when range is not to be normalized
        
        # for synthetic downstream
        # self.layer1 = MonotonicLinear(1, 10, gate_func='relu', indicator=torch.tensor([1]),)
        # self.layer2 = MonotonicLinear(10, 10, gate_func='relu', indicator=torch.tensor([1]*10),)
        # self.layer4 = nn.Linear(10,1) # use only when range is not to be normalized
    #     self.normalized_range = normalized_range
    #     self.max_x = max_x
    #     self.min_x = min_x

    # def normalize(self, x):
    #     if self.normalized_range:
    #         return 2 * (x - self.min_x) / (self.max_x - self.min_x) - 1  # Scale to [-1, 1]
    #     else:
    #         return x 

    # def inverse_normalize(self, x):
    #     if self.normalized_range:
    #         return (x + 1) * (self.max_x - self.min_x) / 2 + self.min_x
    #     else:
    #         return x

    def regularization_loss(self, lambda_reg=[0,0]):
        reg_loss = 0
        # for i, weights in enumerate([self.layer1.weight, self.layer2.weight, self.layer4.weight]):
        for i, weights in enumerate([self.layer1.weight, self.layer4.weight]):
            l2_loss = torch.sum(weights ** 2)  # Standard L2 regularization
            # if i == 1:
            #     l2_loss = torch.sum((weights-1) ** 2)  # Standard L2 regularization
            # else:
            #     l2_loss = torch.sum(weights ** 2)  # Standard L2 regularization
            reg_loss += lambda_reg[i] * l2_loss 
        return reg_loss

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        # use linear output to cover -inf to +inf, otherwise, map to 
        # original range via inverse map.
        self.layer4.weight.data.clamp_(min=0.0)
        x = self.layer4(x) 
        return x

class LinearRegressionNN(nn.Module):
    def __init__(self,normalized_range=False, identity=False):
        super(LinearRegressionNN, self).__init__()
        self.identity = identity
        self.layer1 = MonotonicLinear(1, 10, gate_func='relu', indicator=torch.tensor([1]),)
        self.linear = nn.Linear(10, 1)  # Input size = 1, Output size = 1
        # self.layer1 = MonotonicLinear(1, 10, gate_func='relu', indicator=torch.tensor([1]),)
        # self.linear = nn.Linear(10, 1)  # Input size = 1, Output size = 1
        # self.linear = nn.Linear(1, 1)  # Input size = 1, Output size = 1
        self.normalized_range = normalized_range
        if normalized_range:
            self.tanh = nn.Tanh()

    def forward(self, x):
        if self.normalized_range:
            return self.tanh(self.linear(x))
        else:
            if self.identity: # for the synthetic downstream experiment
                return x
            x = self.layer1(x)
            return self.linear(x)
    
    def regularization_loss(self, desired_weight, desired_bias, lambda_reg):
        weight = self.linear.weight
        bias = self.linear.bias
        reg_loss = lambda_reg * ((weight - desired_weight) ** 2 + (bias - desired_bias) ** 2).sum()
        return reg_loss



class CompositeModel(nn.Module):
    def __init__(self, model1, model2, f1, normalized_range=False, f2_min=None, f2_max=None):
        super(CompositeModel, self).__init__()
        self.model1 = model1  # First model
        self.model2 = model2  # Second model
        self.f1 = f1
        self.normalized_range = normalized_range
        self.f2_min = f2_min
        self.f2_max = f2_max

    def rescale(self, f2_hat):
            return 2 * (f2_hat - self.f2_min) / (self.f2_max - self.f2_min) - 1  # Scale to [-1, 1]

    def forward(self, obs, preds):
        # w_x = self.model1(pred)
        if self.normalized_range:
            out = self.model2(self.rescale(self.f1(w_x)))
        else:
            # print("asdasdasdasd")
            f1_out = self.f1(obs, preds, v=self.model1)
            # print(f1_out.dtype)
            # print(f1_out.shape)
            # print("$$$$$$$$$$$$")
            out = self.model2(f1_out.view(-1,1))
            # out = f1_out.view(-1,1)
        # out = self.f1(w_x)
        return out  # Output of reg_model and transformed w_x


#%% Pipeline for learning toy example test cases
def run(x, f1, f2):
    losses = []
    main_losses = []
    identitiy_penalty_losses = []
    weight1_penalty_losses = []
    weight2_penalty_losses = []

    # Example training loop
    normalized_range = False
    f2_max = None
    f2_min = None
    if normalized_range:
        def normalize(x, min_x, max_x):
            return 2 * (x - min_x) / (max_x - min_x) - 1  # Scale to [-1, 1]
        f2_org = f2(x)
        f2_min = torch.min(f2_org)
        f2_max = torch.max(f2_org)
        f2_normalized = normalize(f2_org, f2_min, f2_max) 
    model_mon = MonotonicNN()  
    model_reg = LinearRegressionNN(normalized_range=normalized_range)
    model = CompositeModel(model_mon, model_reg, f1, normalized_range=normalized_range, f2_max=f2_max, f2_min=f2_min)
    # model = model_ssg
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)#, betas=(0.8, 0.999))#, weight_decay=1e-4)
    mseLoss = nn.MSELoss()

    best_loss = float("inf")  # Initialize best loss
    best_model_state = None  # Placeholder for best model state

    epoch=-1
    loss = torch.tensor(torch.inf)
    # run the fitting while epochs is at least 3000 steps or at most 10000 and loss is not good enough
    max_epoch = 10000
    min_epoch = 3000
    # max_epoch = 1000
    # min_epoch = max_epoch
    while (epoch < max_epoch and loss.item() > 0.01) or epoch < min_epoch:

        epoch += 1
        # w_x = model_ssg(x)
        f1_out, w_x = model(x)
        # f1_out = f1(w_x)
        # main_loss = torch.mean(torch.abs(f1_out - f2(x))) # L1 loss
        # main_loss = torch.mean((f1_out - f2(x))**2) # L2 loss

        if normalized_range:
            main_loss = mseLoss(f1_out, f2_normalized)  # MSE pytorch
        else:
            main_loss = mseLoss(f1_out, f2(x))  # MSE pytorch
        loss = main_loss  

        # Add regularization to keep close to identity
        # identity_penalty = torch.mean((w_x - x)**2)
        identity_penalty = 0.5 * torch.mean(torch.abs(w_x - x)**2)
        loss += identity_penalty

        # to ensure monotonically increasing mapping when using linear output
        # penalize negative weights for the linear output!
        # reg_loss1 = model.model1.regularization_loss(lambda_reg=[0.001, 0.001])
        # loss += 0 if normalized_range else reg_loss1 

        # to ensure identity (w=1, b=0)
        reg_loss2 = model.model2.regularization_loss(desired_weight=1.0, desired_bias=0.0, lambda_reg=0.00)
        loss += reg_loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save the best model state
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model_state = model.state_dict()  # Save the best model parameters

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

        losses.append(loss.item())
        main_losses.append(main_loss.item())
        identitiy_penalty_losses.append(identity_penalty.item())
        # weight1_penalty_losses.append(reg_loss1.item())
        weight2_penalty_losses.append(reg_loss2.item())


    # Load the best model state after training
    model.load_state_dict(best_model_state)
    print(f"Best Loss Achieved: {best_loss:.6f}")

    # plot loss curves
    plt.title('loss curves')
    plt.plot(losses, label='total loss')
    plt.plot(main_losses, label='function space loss')
    plt.plot(identitiy_penalty_losses, label='non-identity penalty (w space loss) L2')
    plt.yscale('log')
    # plt.legend()
    # plt.show()

    plt.title('weight decay losses')
    plt.plot(weight1_penalty_losses,label='first network weights L2')
    plt.plot(weight2_penalty_losses, label='second network weights L2')
    plt.yscale('log')
    plt.legend()
    plt.show()

    # Return the model
    return model

#%%
# def normalize(x, min_x, max_x):
#     return 2 * (x - min_x) / (max_x - min_x) - 1  # Scale to [-1, 1]

# f2_org = f2(x)
# f2_normalized = normalize(f2_org, torch.min(f2_org), torch.max(f2_org)) 

# f1p_res, w_x_plot = model(x)

# plt.plot(f2_normalized)
# plt.plot(f1p_res.detach().numpy())
# plt.show()

#%% toy example test cases

# Case 1 - shift in x
# Case 2 - shift in y
# Case 3 - identity

# Define f1 and f2 (example convex functions)
def f1(w_x):
    # return (w_x)**2             # identity
    # return (2*w_x)**2           # x magnitude 
    # return 5*(w_x)**2           # y magnitude 
    # return (w_x)**2             # x power  
    # return (w_x)**2 - 2         # shift in y (sometimes works sometimes doesn't)
    return (w_x + 3)**2       # shift in x 
    # return 5*(w_x-1)**2+4       # y magnitude, x magnitude, shift x, shift y

def f2(x):
    # return (x)**2              # identity
    # return (x)**2              # x magnitude 
    # return (x)**2              # y magnitude 
    # return (x)**4              # x power
    # return (x)**2 + 1          # shift in y (sometimes works sometimes doesn't)
    return (x - 1)**2        # shift in x 
    # return 2*(1.2*x - 3)**2-5  # y magnitude, x magnitude, shift x, shift y 

x = torch.linspace(-10, 10, 1000).unsqueeze(1)  # Input data
f2_res = f2(x)
f2_min_i = np.argmin(f2_res)
f1_res = f1(x)
f1_min_i = np.argmin(f1_res)

plt.figure(figsize=(6, 6))
plt.plot(x, f2_res, label='f2 (target)')
plt.plot(x[f2_min_i], f2_res[f2_min_i], '*', color='blue')
plt.plot(x, f1(x), label='f1 (source)')
plt.plot(x[f1_min_i], f1_res[f1_min_i], '*', color='orange')
plt.grid(alpha=0.25)
plt.legend()
plt.show()

# f2_org = f2(x)
# def f1_normalized(f1):
#     return normalize(f1(x), torch.min(f2_org), torch.max(f2_org))

x_plot = x 
# model = run(x, f1_normalized, f2)
model = run(x, f1, f2)
# w_x_plot = model(x)
f1p_res, w_x_plot = model(x)

f1p_res = f1p_res.squeeze().detach().numpy()
w_x_plot = w_x_plot.squeeze().detach().numpy()

f2_res = f2(x)
f2_min_i = np.argmin(f2_res)
f1_res = f1(x)
f1_min_i = np.argmin(f1_res)
# f1p_res = f1(w_x_plot)
f1p_min_i = np.argmin(f1p_res)

plt.figure(figsize=(12, 5))
plt.subplot(1,2,1)
plt.plot(x_plot, w_x_plot, label="Learned monotonic function", color='k')
plt.plot(x_plot, x_plot, label="Identity function", linestyle="--", color='k')
plt.plot(x_plot[f1_min_i], w_x_plot[f1_min_i], '*', color='k')
plt.plot(x_plot[f1p_min_i], x_plot[f1p_min_i], '*', color='k', label='global optima')
plt.title("Learned Monotonic Transformation of x")
plt.xlabel("x")
plt.ylabel("w(x)")
plt.legend()
plt.grid(alpha=0.25)

plt.subplot(1,2,2)
plt.title("Function Space")
plt.plot(f2(x), label='f2 (target)', color='blue', linewidth=2)
plt.plot(f1(x), label='f1 (source)', color='orange')
# plt.plot(f1(w_x_plot), label='f1 (weighted source)', color='red', linestyle='--', linewidth=2)
plt.plot(f1p_res, label='f1 (weighted source)', color='red', linestyle='--', linewidth=2)
plt.plot(f2_min_i, f2_res[f2_min_i], '*', color='blue')
plt.plot(f1_min_i, f1_res[f1_min_i], '*', color='orange')
plt.plot(f1p_min_i, f1p_res[f1p_min_i], '*', color='red')
plt.plot([], [], '*', color='k', label='global optima')
plt.xlabel("x")
plt.ylabel("w(x)")
# plt.yscale('log')
plt.legend()
plt.grid(alpha=0.25)
plt.show()

plt.figure(figsize=(12,1))
idx = np.array(list(range(0, len(w_x_plot), 10)))
plt.plot(x[idx], np.zeros(len(x))[idx], "|", markersize=20, label='x')
plt.plot(w_x_plot[idx], np.zeros(len(w_x_plot))[idx], "|", markersize=10, label='w(x)')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


#%% CRPS torch implementation

# crps
def crps_ensemble(measurements, ensemble_predictions, v=None):
    """ inspired by the implementation from properscoring rule package """
    def identity(h):
        return h
    v = identity if v is None else v
    obs = v(measurements)
    preds = v(ensemble_predictions)
    obs_diff = np.abs(obs[:,np.newaxis] - preds)
    pred_diff = np.abs(np.expand_dims(preds, -1) - np.expand_dims(preds, -2))
    # crps = obs_diff.mean(axis=-1) - 0.5 * pred_diff.mean(axis=(-2,-1)) # average over scenarios
    crps = obs_diff.mean(axis=-1) - 0.5 * pred_diff.mean(axis=(-2,-1)) 
    # print('obs diff ', obs_diff.shape)
    # print('obs diff mean', obs_diff.mean(axis=-1).shape)
    # print('pred diff ', pred_diff.shape)
    # print('pred diff mean', pred_diff.mean(axis=-2).shape)
    return crps

def crps_ensemble_torch(measurements, ensemble_predictions, v=None, average=False):
    """ inspired by the implementation from properscoring rule package """
    pred_dim0 = ensemble_predictions.shape[0]
    pred_dim1 = ensemble_predictions.shape[1]
    # print(pred_dim0)
    # print(pred_dim1)
    #
    def identity(h):
        return h
    v = identity if v is None else v
    #
    # weighting functions are applicable only to 1 dimensional arrays and it is 
    # applied element-wise.
    # apply weighting on 1D arrays and turn them back to the original dimension.
    obs_t = v(measurements.view(-1,1)).view(pred_dim0)
    preds_t = v(ensemble_predictions.view(-1,1)).view(pred_dim0, pred_dim1)
    #
    obs_diff = torch.abs(obs_t[:,None] - preds_t)
    pred_diff = torch.abs(torch.unsqueeze(preds_t, -1) - torch.unsqueeze(preds_t, -2))
    # TODO: the average across scenario is a proper case. Otherwise, it is not really proper
    # need to rerun and test everything with average instead of previously run experiments
    # with the non-averaged crps.
    if average:
        crps = obs_diff.mean(axis=-1) - 0.5 * pred_diff.mean(axis=(-2,-1))
    else:
        crps = obs_diff - 0.5 * pred_diff.mean(axis=(-2))
    return crps

def crps_ensemble_torch_wrapper(average=False):
    def crps_ensemble_torch_(measurements, ensemble_predictions, v):
        return crps_ensemble_torch(measurements, ensemble_predictions, v=v, average=average)
    return crps_ensemble_torch_

#%% Prepare chaining functions

np.random.seed(1000)

def chaining(c_type='', **arg):
    if c_type == 'threshold':
        def chaining_(z):
            return v1(z, arg['t'])
    elif c_type == 'interval':
        def chaining_(z):
            return v2(z, arg['a'], arg['b'])
    elif c_type == 'gaussian':
        def chaining_(z):
            return v3(z, arg['t'], arg['mu'], arg['sig'])
    elif c_type == 'sumsigmoids':
        def chaining_(z):
            return sum_of_sigmoid(z, arg['a'], arg['b'], arg['c'], arg['d'])
    elif c_type == 'betafunc':
        def chaining_(z):
            return beta_calib_func(z, arg['b'], arg['a'], arg['m'])
    elif c_type == 'unk': # if unknown return identity
        def chaining_(z):
            return z 
    else:
        assert False, 'invalid chaining type. Available optios: "threshold", "interval", or "gaussian"'
    return chaining_


def plot_chaining_func(z, v, title='', label='', filename_save=''):
    plt.title(title)  if filename_save == '' else None
    plt.plot(z, v, label=label)
    plt.xlabel('$z$')
    plt.ylabel('$\\nu$')
    plt.grid(alpha=0.25)
    # plt.legend() if label != '' else None
    if filename_save:
        plt.savefig(f"figs/synth_reg/chaining_{filename_save}.pdf", format="pdf", bbox_inches = 'tight')
    plt.show()


c_func_args = {'c_type': 'threshold',
                't': 0.5
              }
z = np.linspace(-2, 2, 500)
plot_chaining_func(z, chaining(**c_func_args)(z), 
                   title=f"chaining function t = {c_func_args['t']}", 
                   label=f'$\\nu(z;t={c_func_args["t"]}) = max(z, t)$',
                   filename_save='threshold')


c_func_args = {'c_type': 'interval',
                'a': -0.5,
                'b': 1.5
              }
z = np.linspace(-2, 2, 500)
plot_chaining_func(z,chaining(**c_func_args)(z), 
                   title=f"chaining function interval = [{c_func_args['a']},{c_func_args['b']}]", 
                   label=f'$\\nu(z;a={c_func_args["a"]},b={c_func_args["b"]}) = min(max(z, a), b)$',
                   filename_save='interval')


c_func_args = {'c_type': 'gaussian',
                't': 0.5,
                'mu': 0,
                'sig': 1
               }
z = np.linspace(-2, 2, 500)
plot_chaining_func(z,chaining(**c_func_args)(z), 
                   title=f"chaining function gaussian threshold t = {c_func_args['t']}, mu = {c_func_args['mu']}, sig = {c_func_args['sig']}", 
                   label=f'',
                   filename_save='gaussian')

c_func_args = {'c_type': 'sumsigmoids',
                'a': np.array([[0, 5, 1, 2]]),
                'b': np.array([2, 10, 20, -1]),
                'c': np.array([1, 4, 2, 5]),
                'd': np.array([0, 0, 0, 0])
}
z = np.linspace(-2, 2, 500)
out = chaining(**c_func_args)(z)
plot_chaining_func(z, chaining(**c_func_args)(z), 
                   title=f"chaining function sum of sigmoids 1", 
                   label=f'',
                   filename_save='sumsigmoid_1')

c_func_args = {'c_type': 'sumsigmoids',
                'a': np.array([[20, 18, 22, 17, 13]]),
                'b': np.array([ -7,   3, 14, -10,  16]),
                'c': np.array([15, 11, 19, 12, 9]),
                'd': np.array([0])
}
z = np.linspace(-2, 2, 500)
plot_chaining_func(z, chaining(**c_func_args)(z), 
                   title=f"chaining function sum of sigmoids 2", 
                   label=f'',
                   filename_save='sumsigmoid_2')

c_func_args = {'c_type': 'betafunc',
                'a': 0.5,
                'b': 0.2,
                'm': 0.75
}
z = np.linspace(0, 1, 500)
plot_chaining_func(z, chaining(**c_func_args)(z), 
                   title=f"chaining function beta", 
                   label=f'',
                   filename_save='beta')

#%%
# select the chaining function and data source

# choose chaining function by uncommenting any of the following: 
c_func_args = {'c_type': 'threshold',
                't': 0.5
              }

# c_func_args = {'c_type': 'interval',
#                 'a': -0.5,
#                 'b': 1.5
#               }

# c_func_args = {'c_type': 'gaussian',
#                 't': 0.5,
#                 'mu': 0,
#                 'sig': 1
#                }

# c_func_args = {'c_type': 'sumsigmoids',
#                 'a': np.array([[0, 5, 1, 2]]),
#                 'b': np.array([2, 10, 20, -1]),
#                 'c': np.array([1, 4, 2, 5]),
#                 'd': np.array([0, 0, 0, 0])
# }

# c_func_args = {'c_type': 'sumsigmoids',
#                 'a': np.array([[20, 48, 22, 37, 13]]),
#                 'b': np.array([ 47,   3,  34, -40,  46]),
#                 'c': np.array([25, 21, 39, 32, 19]),
#                 'd': np.array([-25])
# }

# c_func_args = {'c_type': 'sumsigmoids',
#                 'a': np.array([[20, 18, 22, 17, 13]]),
#                 'b': np.array([ -7,   3, 14, -10,  16]),
#                 'c': np.array([15, 11, 19, 12, 9]),
#                 'd': np.array([0])
# }

# c_func_args = {'c_type': 'betafunc',
#                 'a': 0.5,
#                 'b': 0.2,
#                 'm': 0.75
# }


# Select experiment and data source. Three options available:
# 0) downstream from a real downstream formulation and real demand forecast and prices
# 1) syntheitc downstream from a real downstream formulation but with a synthetic demand forecast
# 2) synthetic regression data produced by a regression model based on a synthetic regression dataset
# 3) synthetic regression model output using a random normal distribution

# load_data_from_file = True # option 1 and 2 are stored in the file
# downstream_task_synth = True # if option 1 is needed
# downstream_task_real = True # if option 1 is needed

experiment_types = ["real_downstream_newsvendor",  # newsvendor with real data (predictive distribution on real data)
                    "synth_downstream_newsvendor", # newsvendor with synthetic data (predictive distribution from a fixed distribution)
                    "synth_downstream_sanity_check", # crps vs twcrps: synthetic downstream eval via a twcrps with known weight function (predictive distribution from a synth regression data)
                    "synth_downstream_sanity_check2", # crps vs twcrps: synthetic downstream eval via a twcrps with known weight function (predictive distribution from a fixed distribution)
                    "synth_downstream_sanity_check3" # crps vs logloss: synthetic downstream eval via logloss while upstream eval is crps
                    ]

downstream_newsvendor = False # customized plots for this experiment
experiment_type = experiment_types[0]

print("Experiment: ", experiment_type)
#
if experiment_type == experiment_types[0]: # real downstream data
    downstream_newsvendor = True # this should not be changed!
    save_fig_path = "tsukiji_seafood"
    c_func_args = {'c_type': 'unk'} 
    # load downstream data from file
    # directory = "downstream_realdata"
    #
    dataset_names = [
                    "Bluefin_Fresh_JapaneseFleet",
                    "Bluefin_Fresh_ForeignFleet",
                    "Bluefin_Frozen_UnknownFleet",
                    "Southern_Fresh_UnknownFleet",
                    "Southern_Frozen_UnknownFleet",
                    "Bigeye_Fresh_UnknownFleet"
    ]
    dataset_name = dataset_names[3]
    constant_price = False
    constant_price_str = "const" if constant_price else "vary"
    save_fig_id = f"{dataset_name}_price-{constant_price_str}"
    
    # set the type of prediction compared to its test time distribution
    def load_files(directory):
        measurements = None
        ensemble_predictions = None
        critical_quantile_values = None
        downstream_values_p = None
        downstream_values_o = None
        for file_type in ["obs", "preds", "critical_qs", "profits_pred", "profits_obs"]:
            filename = f"tuna_{dataset_name}_price-{constant_price_str}_{file_type}.npy"  # use for real downstream data from kaggle tsukiji dataset
            filepath = os.path.join(directory, filename)
            if file_type == "obs":
                measurements = np.load(filepath)
            elif file_type == "preds":
                ensemble_predictions = np.load(filepath)
            elif file_type == "critical_qs":
                critical_quantile_values = np.load(filepath)
            elif file_type == "profits_pred": # profits based on predictive distribution
                downstream_values_p = np.load(filepath)
            elif file_type == "profits_obs": # profits based on observations
                downstream_values_o = np.load(filepath)
            else:
                assert False, "filetype not permissible."
        return measurements, ensemble_predictions, critical_quantile_values, downstream_values_p, downstream_values_o
    
    # alignment data (holdout/validation from the predictive task)
    measurements, ensemble_predictions, critical_quantile_values, downstream_values_p, downstream_values_o = load_files(directory = "tuna data/profits/validation")
    # downstream_values = None
    # test data
    measurements_te, ensemble_predictions_te, critical_quantile_values_te, downstream_values_p_te, downstream_values_o_te = load_files(directory = "tuna data/profits/test")
    print("loaded downstream files.")
elif experiment_type == experiment_types[1]: # synthetic downstream data
    downstream_newsvendor = True
    save_fig_path = "synth_seafood"
    # c_func_args = {'c_type': 'unknown'} # FIXME: why did that I put this? causes assert False
    # load downstream data from file
    directory = "downstream_data"
    measurements = None # TODO: rename to observations 
    ensemble_predictions = None
    downstream_values = None
    # set the type of prediction compared to its test time distribution
    # pred_type = "optimal" 
    pred_type = "nonoptimal" 
    for file_type in ["obs", "preds", "profits_pred", "profits_obs"]:
        filename = f"seafood_weibull_{pred_type}_{file_type}.npy"  # use for synthetic downstream data
        filepath = os.path.join(directory, filename)
        if file_type == "obs":
            measurements = np.load(filepath)
        elif file_type == "preds":
            ensemble_predictions = np.load(filepath)
        elif file_type == "profits_pred":
            downstream_values_p = np.load(filepath)
        elif file_type == "profits_obs":
            downstream_values_o = np.load(filepath)
        else:
            assert False, "filetype not permissible."
    print("loaded downstream files.")
elif experiment_type == experiment_types[2]: # synthetic regression data
        save_fig_path = "synth_reg"
        # load sythetic regression data from file
        directory = "synthetic_regression_data"
        data_type = "1"  # 1 sinusoidal form, 2 exponential form
        noise_type = "2" # 1 homoskedastic, 2 heteroskedastic
        data_type_str = 'sinusoidal' if data_type == "1" else 'exponential'
        noise_type_str = 'homo' if noise_type == "1" else 'hetero'
        save_fig_id = f"data-{data_type_str}_noise-{noise_type_str}"
        # select prediction type:
        def load_files(filename, suffix, set_name=''):
            """ load synthetic regression data from file 
                set_name default is whole dataset. options: val, test
                suffix is optimal, biased, highvar, or lowvar
            """
            # filename = f"data{data_type}{noise_type}_model_optimal.npy" 
            # filename = f"data{data_type}{noise_type}_model_biased.npy" 
            # filename = f"data{data_type}{noise_type}_model_highvar.npy" 
            # filename = f"data{data_type}{noise_type}_model_lowvar.npy" 
            if set_name != '':
                filename = f"{filename}_{suffix}_{set_name}.npy" 
            else:
                filename = f"{filename}_{suffix}.npy" 
            filepath = os.path.join(directory, filename)
            arr = np.load(filepath)
            X_tmp = arr[:,0]
            measurements = arr[:,1]
            ensemble_predictions = arr[:,2:]
            n_observations = measurements.shape[0]
            n_samples = ensemble_predictions.shape[1]
            return X_tmp, measurements, ensemble_predictions, n_observations, n_samples

        # X_tmp, measurements, ensemble_predictions, n_observations, n_samples = load_files(f"data{data_type}{noise_type}_model", suffix="optimal", set_name='')
        X_tmp, measurements, ensemble_predictions, n_observations, n_samples = load_files(f"data{data_type}{noise_type}_model", suffix="optimal", set_name='val')
        X_tmp_te, measurements_te, ensemble_predictions_te, _, _ = load_files(f"data{data_type}{noise_type}_model", suffix="optimal", set_name='test')

        idx = np.argsort(X_tmp)
        plt.figure(figsize=(7,5))
        plt.title("Synthetic Regression Data")
        plt.plot(X_tmp[idx], ensemble_predictions[idx], '.', color='red', alpha=0.1)
        plt.plot(X_tmp[idx], measurements[idx], '.', label='Observation $y$')
        plt.plot([],[], '-', color='red', alpha=0.5, label='Prediction $x \sim F_X$')
        plt.plot(X_tmp[idx], ensemble_predictions[idx].mean(axis=1), '-', color='black', label='Mean prediction $\mathbb{E}[X]$')
        plt.xlabel('feature')
        plt.ylabel('target')
        plt.grid(alpha=0.25)
        plt.legend()
        plt.savefig("figs/synthetic_regdata.png", dpi=300, bbox_inches='tight')
        plt.show()
elif experiment_type == experiment_types[3]: # synthetic regression output data -- no regression model involved. 
    save_fig_path = "synth_noreg"
    # generate random samples for the target variable and predictive distribution from a normal distribution
    # the predictive distribution represents output of a regression model without having an actual regression model
    n_observations = 1000
    n_samples = 100 # scenarios
    #
    measurements = np.random.normal(loc=0, scale=1, size=(n_observations))
    ensemble_predictions = np.random.normal(loc=2, scale=1, size=(n_observations, n_samples))
    #
    # n_quantiles = 100
    # quantiles = np.linspace(0.01,1,n_quantiles,endpoint=False)
    # quantile_predictions = np.swapaxes(np.quantile(ensemble_predictions, quantiles, axis=0), 0, 1)
elif experiment_type == experiment_types[4]: 
    # synthetic regression output data -- no regression model involved. 
    # downstream evaluation via logloss while upstream is twcrps
    save_fig_path = "synth_noreg_logloss"
    n_observations = 1000
    n_samples = 100 # scenarios
    #
    # measurements = np.random.normal(loc=0, scale=1, size=(n_observations))
    # ensemble_predictions = np.random.normal(loc=2, scale=1, size=(n_observations, n_samples))
    # weibull distribution
    # shape = 2
    # true_scale = 50 # Slightly scaled true Weibull distribution
    # true_shift = 20  # Slightly shifted true Weibull distribution
    # scale = 100
    # shift = 20
    # measurements = true_scale * np.random.weibull(a=shape, size=(n_observations,)) + true_shift
    # ensemble_predictions = scale * np.random.weibull(a=shape, size=(n_observations, n_samples)) + shift
    # skwed normal
    from scipy.stats import skewnorm
    skewness = 3
    measurements = skewnorm.rvs(a=-1, loc=0, scale=1, size=n_observations)
    ensemble_predictions = skewnorm.rvs(a=skewness, loc=2, scale=1.5, size=(n_observations, n_samples))

else:
    assert False, f"Experiment type '{experiment_type}' does not exist!"

# plot the histogram of samples from the target variable and predictive distribution.
plt.hist(measurements, density=True, label='observations')
plt.hist(ensemble_predictions.flatten(),alpha=0.5, density=True, label='predictions')
plt.legend()
plt.show()

print("####")

crps_ref = ps.crps_ensemble(measurements, ensemble_predictions) # reference is used from the properscoring package to verify implementation
crps = crps_ensemble(measurements, ensemble_predictions, v=None) # none-weightd torch implementation 
crps_te = crps_ensemble(measurements_te, ensemble_predictions_te, v=None) 
crps_weighted = crps_ensemble(measurements, ensemble_predictions, v=chaining(**c_func_args)) # weighted torch implementation
crps_weighted_te = crps_ensemble(measurements_te, ensemble_predictions_te, v=chaining(**c_func_args)) # weighted torch implementation

# to verify results compared to the reference
print(crps_ref.mean())
print(crps.mean())
print(crps_weighted.mean())

# plot different weighted crps compared to its none-weighted version for diagnosis purposes
crps_vs = []
for t_ in np.linspace(-3,3,10):
    c_func_args_tmp = {'c_type': 'threshold',
                        't': t_
                    }
    crps_v = crps_ensemble(measurements, ensemble_predictions, v=chaining(**c_func_args_tmp))
    crps_vs.append(crps_v.mean())

plt.plot(crps_vs, label='twCRPS (v)')
plt.hlines(crps.mean(), 0, len(crps_vs), color='k', label='CRPS')
plt.legend()
plt.show()

# crps_ref = ps.crps_ensemble(measurements[:n_observations//2], measurements[n_observations//2:,np.newaxis])
# crps_ref = ps.crps_ensemble(measurements[:n_observations//2], measurements[n_observations//2:,np.newaxis])
# print(crps_ref.mean())

#%% Pipeline for learning CRPS alignment 

if downstream_newsvendor:
    # this is a hack to bring the downstream scores to more or less similar scale as the upstream scores
    # TODO: try to resolve this without minmaxscaler. The alignment pipeline should be able to handle this rescaling!
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(np.min(crps),np.max(crps)))

    # plot downstream scores obtained by evaluating the downstream objective function on 
    # the samples from the predictive distribution and samples from the test samples.
    plt.title('profit distribution based on predictive and observation samples')
    plt.hist(downstream_values_p.flatten(), label='profits from predcitive samples', density=True)
    plt.hist(downstream_values_o.flatten(), label='profits from observation samples', density=True, alpha=.5)
    plt.legend()
    plt.show()

    # scale transform of the downstream values 
    # downstream_values_t = scaler.fit_transform(-downstream_values_p.mean(axis=1, keepdims=True))
    downstream_values_t = scaler.fit_transform(-downstream_values_o)
    downstream_values_te_t = scaler.transform(-downstream_values_o_te)
    # downstream_values_t = -downstream_values_o

    # similar plot as last one but now with the rescaled scores obtained from the test samples
    plt.title('profit distribution after scaling')
    plt.hist(downstream_values_o.flatten())
    plt.hist(downstream_values_t.flatten())
    plt.show()

    # downstream_values_t = scaler.fit_transform(-downstream_values_p.mean(axis=1, keepdims=True))
    # plt.hist(downstream_values_p.mean(axis=1))
    # plt.hist(downstream_values_t.flatten())
    # plt.show()

    # the rescaled scores should be now more comparable in range and scale with the upstream scores
    plt.title('expected profit across observations - profits.mean(axis=1)')
    # plt.hist(crps, density=True)
    plt.hist(downstream_values_t, density=True)
    plt.hist(downstream_values_te_t, density=True)
    plt.show()


plt.title("Distribution of Upstream and Downstream Scores")
plt.hist(crps, label='upstream scores ≡ unweighted CRPS', density=True)
plt.hist(crps_te, label='upstream scores ≡ unweighted CRPS (test)', density=True)
if downstream_newsvendor:
    plt.hist(downstream_values_t.flatten(), alpha=0.5, label='downstream scores', density=True)
    plt.hist(downstream_values_te_t.flatten(), alpha=0.5, label='downstream scores (test)', density=True)
else:
    plt.hist(crps_weighted, alpha=0.5, label='downstream scores ≡ weighted CRPS', density=True)
    plt.hist(crps_weighted_te, alpha=0.5, label='downstream scores ≡ weighted CRPS (test)', density=True)
    #
    # parametric logscore
    # logscores = sr.logs_normal(measurements, 2, 1)
    # plt.hist(logscores, alpha=0.7, label='downstream scores ≡ Logloss', density=True)
    #
    # nonparameteric logscore
    # from sklearn.neighbors import KernelDensity
    # logscores_kde = []
    # for i in range(n_observations):
    #     # kde = gaussian_kde(ensemble_predictions[:,i])
    #     # score = kde.logpdf(measurements[i])
    #     # sklearn
    #     kde = KernelDensity(kernel='gaussian', bandwidth='scott')
    #     kde.fit(ensemble_predictions[i][:,np.newaxis])
    #     #
    #     score = -kde.score_samples(measurements[i].reshape(-1, 1))
    #     logscores_kde.append(score)
    # logscores_kde = np.array(logscores_kde)
    # plt.hist(logscores_kde, alpha=0.7, label='downstream scores ≡ Logloss KDE', density=True)

plt.legend()
plt.show()

#%% Run the alignment pipeline

losses = []
main_losses = []
identitiy_penalty_losses = []
weight1_penalty_losses = []
weight2_penalty_losses = []


# Example training loop
normalized_range = False
f2_max = None
f2_min = None
if normalized_range:
    def normalize(x, min_x, max_x):
        return 2 * (x - min_x) / (max_x - min_x) - 1  # Scale to [-1, 1]
    f2_org = f2(x)
    f2_min = torch.min(f2_org)
    f2_max = torch.max(f2_org)
    f2_normalized = normalize(f2_org, f2_min, f2_max) 


scoring_rule = crps_ensemble_torch_wrapper(average = True)
# Model definition
model_mon = MonotonicNN()  
if not downstream_newsvendor:
    deactivate_output_transform = True
    print("model 2 (output transform) deactivated.")
else:
    deactivate_output_transform = False
    print("model 2 (output transform) active.")
model_reg = LinearRegressionNN(normalized_range=normalized_range, identity=deactivate_output_transform)
model = CompositeModel(model_mon, model_reg, scoring_rule, normalized_range=normalized_range, f2_max=f2_max, f2_min=f2_min)
# model = model_ssg
optimizer = torch.optim.Adam(model.parameters(), lr=0.04, weight_decay=1e-5)#, betas=(0.8, 0.999))#, weight_decay=1e-4)
mseLoss = nn.MSELoss()

best_loss = float("inf")  # Initialize best loss
best_model_state = None  # Placeholder for best model state
epoch=-1
loss = torch.tensor(torch.inf)
# run the fitting while epochs is at least 3000 steps or at most 10000 and loss is not good enough
# max_epoch = 1000
min_epoch = 50
max_epoch = 1000 if experiment_type == experiment_types[0] else 300
min_epoch = max_epoch
obs = torch.from_numpy(measurements).float().contiguous()
preds = torch.from_numpy(ensemble_predictions).float().contiguous()

obs_te = torch.from_numpy(measurements_te).float().contiguous()
preds_te = torch.from_numpy(ensemble_predictions_te).float().contiguous()
# crps_toch = crps_ensemble_torch(obs, preds, v=chaining)
# print(crps_torch.shape)
if not downstream_newsvendor:
    # TODO: put a switch between logscore and crps based on the experiment
    f2_out = torch.from_numpy(crps_weighted).float()
    f2_out_te = torch.from_numpy(crps_weighted_te).float()
    # f2_out = torch.from_numpy(logscores).float()
    # f2_out = torch.from_numpy(logscores_kde.flatten()).float()
else:
    f2_out = torch.from_numpy(downstream_values_t).float()
    f2_out_te = torch.from_numpy(downstream_values_te_t).float()

# print(f2_out.shape)
while (epoch < max_epoch and loss.item() > 0.01) or epoch < min_epoch:

    epoch += 1
    # w_x = model_ssg(x)
    f1_out = model(obs, preds)
    # print(f1_out.shape)
    # f1_out = f1(w_x)
    # main_loss = torch.mean(torch.abs(f1_out - f2(x))) # L1 loss
    # main_loss = torch.mean((f1_out - f2(x))**2) # L2 loss

    if normalized_range:
        main_loss = mseLoss(f1_out, f2_normalized)  # MSE pytorch
    else:
        main_loss = mseLoss(f1_out, f2_out.view(-1,1))  # MSE pytorch
    loss = main_loss  

    # Add regularization to keep close to identity
    # identity_penalty = torch.mean((w_x - x)**2)
    # identity_penalty = 0.1 * torch.mean(torch.abs(w_x - x)**2)
    # loss += identity_penalty

    # to ensure monotonically increasing mapping when using linear output
    # penalize negative weights for the linear output!
    # reg_loss1 = model.model1.regularization_loss(lambda_reg=[0,0])#[0.005, 0.0001])
    # reg_loss1 = model.model1.regularization_loss(lambda_reg=[0.005, 0.005, 0.0001])

    # loss += 0 if normalized_range else reg_loss1 
    # loss += 0 # reg_loss1 

    # to ensure identity (w=1, b=0)
    # reg_loss2 = model.model2.regularization_loss(desired_weight=0.0, desired_bias=0.0, lambda_reg=1e-4)
    # reg_loss2 = model.model2.regularization_loss(desired_weight=1.0, desired_bias=0.0, lambda_reg=1e-4)
    # loss += 0
    # loss += reg_loss2

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Save the best model state
    if loss.item() < best_loss:
        best_loss = loss.item()
        best_model_state = model.state_dict()  # Save the best model parameters

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

    losses.append(loss.item())
    main_losses.append(main_loss.item())
    # identitiy_penalty_losses.append(identity_penalty.item())
    # weight1_penalty_losses.append(reg_loss1.item())
    # weight2_penalty_losses.append(reg_loss2.item())


# Load the best model state after training
model.load_state_dict(best_model_state)
print(f"Best Loss Achieved: {best_loss:.6f}")

# plot loss curves
plt.title('loss curves')
plt.plot(losses, label='total loss')
plt.plot(main_losses, label='function space loss')
plt.plot(identitiy_penalty_losses, label='non-identity penalty (w space loss) L2')
# plt.yscale('log')
# plt.legend()
# plt.show()

# plt.title('weight decay losses')
plt.plot(weight1_penalty_losses,label='first network weights L2')
plt.plot(weight2_penalty_losses, label='second network weights L2')
plt.yscale('log')
plt.legend()
plt.savefig(f'figs/{save_fig_path}/{save_fig_id}_result_loss_cruves_ctype_{c_func_args["c_type"]}.pdf', dpi=300, bbox_inches='tight')
plt.show()


#%%

test_set = True
if test_set:
    set_str = "test"
    f2_hat = model(obs_te, preds_te) # currently the model only spits out the monotonic part without the second part
    score_hat = f2_hat.detach().numpy()#.flatten()
    score = f2_out_te.detach().numpy()#.flatten()
    score_nonaligned = crps_te
else:
    set_str = "val"
    f2_hat = model(obs, preds) # currently the model only spits out the monotonic part without the second part
    score_hat = f2_hat.detach().numpy()#.flatten()
    score = f2_out.detach().numpy()#.flatten()
    score_nonaligned = crps

# score_transformed = scaler.inverse_transform(score)
# score_hat_transformed = scaler.inverse_transform(score_hat)
# score_hat = scaler.inverse_transform(score_hat)
# score = scaler.inverse_transform(score)

# plt.title("Distribution of crps scores")
# plt.hist(crps.flatten(), label='f1 (upstream obj ≡ unweighted CRPS)')
# plt.hist(score, label='f2 (downstream obj ≡ weighted CRPS)')
# plt.hist(score_hat, alpha=0.5, label='f2_hat (estimated downstream obj ≡ learned weighted CRPS)')
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3) )
# plt.show()
show_plot_titles = False  # for paper do not show titles
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({
    'font.size': 12,          # Base font size
    'axes.labelsize': 14,      # Axis labels
    'axes.titlesize': 16,      # Title size
    'xtick.labelsize': 11,     # X-axis tick labels
    'ytick.labelsize': 11,     # Y-axis tick labels
    'legend.fontsize': 12,     # Legend font size
    'figure.titlesize': 18,    # Overall figure title size
    'lines.linewidth': 2.25,      # Line thickness
    'lines.markersize': 7      # Marker size
})

# averaged over the scenarios
# plt.figure(figsize=(7,5))
fig, axes = plt.subplots(2,1, sharex=False, height_ratios=[5,1], figsize=(7,7),)
plt.subplots_adjust(hspace=0.2)
axes[0].set_title("Distribution of Scores and Alignment Errors") if show_plot_titles else None
axes[0].hist(crps, color='orange', label='$s^u$ ≡ CRPS', density=True)
# plt.hist(score.reshape(n_observations, n_samples).mean(axis=1), label='f2 (downstream obj ≡ weighted CRPS)')
# plt.hist(score_hat.reshape(n_observations, n_samples).mean(axis=1), alpha=0.5, label='f2_hat (estimated downstream obj ≡ learned weighted CRPS)')
axes[0].hist(score, alpha=0.75, label='$s^d$' if downstream_newsvendor else '$s^d$ ≡ twCRPS', density=True)
axes[0].hist(score, histtype='step', color='blue', linewidth=2, density=True)
axes[0].hist(score_hat, alpha=1.0, linewidth=2, linestyle="--", color='red', histtype='step', label= '$\hat{s}^d$' if downstream_newsvendor else '$\hat{s}^d$ ≡ learned twCRPS', density=True)
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3) )
axes[0].grid(alpha=0.25)
axes[0].set_xlabel(" ")
# plt.ylabel("Density")
axes[0].legend()
# plt.savefig(f'figs/{save_fig_path}/scores_distribution.pdf', dpi=300, bbox_inches='tight')
# plt.show()

error_s = (score.flatten() - score_hat.flatten())
# axes[1].figure(figsize=(7,1))
# title_str = f'Distribution of $(s^d-\hat{{s}}^d)$ with $MAE(s^d, \hat{{s}}^d)={np.abs(error_s).mean():2.3f}, MSE(s^d, \hat{{s}}^d)={(error_s**2).mean():2.3f}$'
title_str = f'Distribution of $(s^d-\hat{{s}}^d)$ with $MAE(s^d, \hat{{s}}^d)={np.abs(error_s).mean():2.3f}$'
# axes[1].set_title(title_str)
axes[1].hist(error_s, density=True, color='grey', label='$(s^d-\hat{{s}}^d)$')
heights = [patch.get_height() for patch in axes[1].patches]
axes[1].vlines(np.abs(error_s).mean(), min(heights), max(heights), label=f'$MAE(s^d, \hat{{s}}^d)={np.abs(error_s).mean():2.2f}$')
axes[1].grid(alpha=0.25)
axes[1].legend()
plt.savefig(f'figs/{save_fig_path}/{save_fig_id}_result_scores_and_error_distributions_ctype_{c_func_args["c_type"]}_{set_str}.pdf', dpi=300, bbox_inches='tight')
plt.show()


#%%

def plot_alignment_curve(score_hat, score, title=''):
    # sorted_idx = np.argsort(score.flatten())
    sorted_idx = np.argsort(score_hat.flatten())
    plt.figure(figsize=(7,7))
    plt.title(title)
    plt.plot(score_hat, score, '.', markersize=10) 
    # plt.plot(score_nonaligned, score, '.', markersize=10) # no trivial to plot
    # plt.plot(score[sorted_idx], score[sorted_idx], '--', color='k', label='perfect alignment') 
    # mid = (score.max() - score.min())/2
    # lower_half = min(mid-score.min(), score.min())
    # upper_half = max(mid+score.max(), score.max())

    mid = (score.min() + score.max() + score_hat.min() + score_hat.max()) / 4
    range_data = max(score.max() - score.min(), score_hat.max() - score_hat.min())
    
    # Define axis limits based on the center and range
    lower_half = mid - range_data / 2
    upper_half = mid + range_data / 2

    xx = np.linspace(lower_half, upper_half, 100)
    plt.plot(xx, xx, '--', color='k', label='Perfect alignment') 

    # plt.xlim([lower_half, upper_half])
    # plt.ylim([lower_half, upper_half])
    plt.xlabel('estimated downstream score $\\hat{s}^d$')
    plt.ylabel('downstream score $s^d$')
    for s_hat, s in zip(score_hat[sorted_idx,0], score[sorted_idx,0]):
        x = s_hat
        y_min, y_max = (s, s_hat) if s_hat > s else (s_hat, s)
        plt.vlines(x, y_min, y_max, color='red', linewidth=1)
    plt.vlines([],[],[], color='red', linewidth=1, label='Error')
    #
    if (experiment_type == experiment_types[0]) or (experiment_type == experiment_types[1]):
        avgscore_label = f'$\\bar{{s}}^d$ = {score.mean()/1e6:2.2f} Million € (Avg net profit)' 
    else:
        avgscore_label = f'$\\bar{{s}}^d$ = {score.mean():2.2f}' 
    plt.hlines(score.mean(),lower_half, upper_half, linewidth=1, label=avgscore_label)
    if (experiment_type == experiment_types[0]) or (experiment_type == experiment_types[1]):
        mae = mean_absolute_error(score, score_hat)
        plt.hlines(mae, lower_half, upper_half, color='orange', linewidth=1, label=f'$MAE(s^d, \hat{{s}}^d)$={mae/1e6:2.2f} Million €')
    else:
        mae = mean_absolute_error(score, score_hat)
        plt.hlines(mae, lower_half, upper_half, color='orange', linewidth=1, label=f'$MAE(s^d, \hat{{s}}^d)$={mae:2.2f}')
    #
    plt.legend()
    plt.grid(alpha=0.25)
    plt.savefig(f'figs/{save_fig_path}/{save_fig_id}_result_scores_alignment_curve_ctype_{c_func_args["c_type"]}_{set_str}.pdf', dpi=300, bbox_inches='tight')
    plt.show()


c = 1e6 if experiment_type == "real_downstream" else 1
if downstream_newsvendor:
    score_inv = -scaler.inverse_transform(score) / c 
    score_hat_inv = -scaler.inverse_transform(score_hat) / c
else:
    score_inv = score.reshape(-1,1)
    score_hat_inv = score_hat.reshape(-1,1)

# plot_alignment_curve(score_hat, score, title='Downstream Score (scaled €) versus its Estimation')
if experiment_type == experiment_types[0]:
    plot_title = 'Downstream Score (Million €) versus its Estimation'
elif experiment_type == experiment_types[1]:
    plot_title = 'Downstream Score (€) versus its Estimation'
else:
    plot_title = 'Downstream Score versus its Estimation'
plot_title='' # for paper do not show titles. add a global parameter for this
plot_alignment_curve(score_hat_inv, score_inv, title=plot_title)

# for diagnosis of the inverse transform
if downstream_newsvendor:
    plt.title('Distirbution of Scores (diagnostic)')
    if test_set:
        plt.boxplot([score_inv.flatten(), downstream_values_o_te.flatten()/ c, score_hat_inv.flatten()])
    else:
        plt.boxplot([score_inv.flatten(), downstream_values_o.flatten()/ c, score_hat_inv.flatten()])
    plt.xticks(range(1,4), ['rescaled target', 'original target', 'estimated target' ])
    plt.savefig(f'figs/{save_fig_path}/{save_fig_id}_result_scores_boxplot_ctype_{c_func_args["c_type"]}_{set_str}.pdf', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_alignment(score, score_hat, print_output=True, eval_name='aligned'):
    r2  = r2_score(score, score_hat)
    mae = mean_absolute_error(score, score_hat)
    mse = mean_squared_error(score, score_hat) 
    mape = mean_absolute_percentage_error(score, score_hat)*100
    def symmetric_mean_absolute_percentage_error(y, yhat):
        return 100/len(y) * np.sum(2 * np.abs(yhat - y) / (np.abs(y) + np.abs(yhat)))
    smape = symmetric_mean_absolute_percentage_error(score, score_hat)
    res = kendalltau(score, score_hat)
    # res = kendalltau(score.flatten(), score.flatten())
    lines = [
        f"MAE: {mae:2.2f}",
        f"MAE/1e6: {mae/1e6:2.2f}",
        f"MSE: {mse:2.2f}",
        f"RMSE: {np.sqrt(mse):2.2f}",
        f"RMSE/1e6: {np.sqrt(mse/1e6):2.2f}",
        f"MAPE: {mape:2.2f}%",
        f"SMAPE: {smape:2.2f}%",
        f"R2: {r2:2.2f}",
        f"kendall-tau: {res.statistic:2.2f}  p-value: {res.pvalue}"
    ]
    evalmetrics_file = f'figs/{save_fig_path}/{save_fig_id}_result_evalmetrics_{eval_name}_ctype_{c_func_args["c_type"]}_{set_str}.txt' 
    with open(evalmetrics_file, 'w') as f:
        for l in lines:
            print(l, file=f) 
            if print_output:
                print(l)

# metrics on the original scale of the scores
evaluate_alignment(score_inv.flatten(), score_nonaligned.flatten(), eval_name='nonaligned')
print("-")
evaluate_alignment(score_inv.flatten(), score_hat_inv.flatten(), eval_name='aligned')
# evaluate_alignment(score.flatten(), score_hat.flatten())

#%%
# averaged over the scenarios
# error_s = error_s.reshape(n_observations, n_samples).mean(axis=1)
# plt.figure(figsize=(6,1))
# plt.title(f'score error - mae: {np.abs(error_s).mean():2.4f}, mse: {(error_s**2).mean():2.4f}')
# plt.hist(error_s)
# plt.show()

if downstream_newsvendor:
    # in case of a downstream task we do not know the true weighting
    z = np.linspace(0, 800, 100)
    chaining_hat = model.model1(torch.from_numpy(z).float().unsqueeze(1)).squeeze().detach().numpy()
    error_v = np.abs(chaining(**c_func_args)(z) - chaining_hat)
    step_idx = np.argmax(chaining_hat) 
    # plt.figure(figsize=(7,5))
    plt.figure(figsize=(7,7))
    plt.title(f'Chaining function - z_star={z[step_idx]:2.3f}') if show_plot_titles else None
    plt.plot(z, chaining_hat, label='Estimated chaining $\\hat{v}$', linestyle='-', color='red')
    # plt.plot(z[step_idx], chaining_hat[step_idx], '.', color='red')
    plt.legend()
    plt.grid(alpha=0.25)
    plt.xlabel('Demand (tons)')
    plt.savefig(f'figs/{save_fig_path}/{save_fig_id}_result_chaining_{set_str}.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    w_hat = np.diff(chaining_hat, 1)
    peak_idx = np.argmax(w_hat[1:])
    # plt.figure(figsize=(7.3,5))
    plt.figure(figsize=(7,7))
    plt.title(f'Weighting function - z_star={z[:-1][peak_idx]:2.3f}') if show_plot_titles else None
    plt.plot(z[:-1], w_hat, label='$d\\hat{v} = \\hat{w}$', color='red', linestyle='-')
    # plt.plot(z[:-1][peak_idx], w_hat[peak_idx], '.', color='red')
    plt.grid(alpha=0.25)
    plt.xlabel('Demand (tons)')
    plt.legend()
    plt.savefig(f'figs/{save_fig_path}/{save_fig_id}_result_weighting_{set_str}.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # plot PDF and CDF of critical values
    # plt.title('Critical Quantile Values PDF')
    # plt.hist(critical_quantile_values)
    # plt.title('Tuna demand (tonne)')
    # plt.show()

    plt.figure(figsize=(7,7))
    plt.title('Critical Quantile Values PDF')
    val, bin_edge = np.histogram(critical_quantile_values, bins=20)
    plt.plot(bin_edge[:-1] + (bin_edge[1:]-bin_edge[:-1])/2, val)
    plt.title('Tuna demand (tonne)')
    plt.savefig(f'figs/{save_fig_path}/{save_fig_id}_result_criticalq_pdf_{set_str}.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    q_vals_sorted = np.sort(critical_quantile_values)
    p = 1. * np.arange(len(critical_quantile_values)) / (len(critical_quantile_values)-1)

    plt.figure(figsize=(7,7))
    plt.title('Critical Quantile Values CDF')
    plt.plot(q_vals_sorted, p)
    plt.title('Tuna demand (tonne)')
    plt.savefig(f'figs/{save_fig_path}/{save_fig_id}_result_criticalq_cdf_{set_str}.pdf', dpi=300, bbox_inches='tight')
    plt.show()# 

    # from sklearn.preprocessing import MinMaxScaler

    # scaler = MinMaxScaler()
    # chaining_hat_t = scaler.fit_transform(chaining_hat.reshape(-1,1))
    # w_hat_t = np.diff(chaining_hat_t.flatten(), 1)
    # plt.plot(z, chaining_hat_t, label='Estimated chaining $\\hat{v}$', linestyle='-', color='red')
    # plt.plot(q_vals_sorted, p)
    # plt.show()

    # plt.plot(z[:-1], w_hat_t, label='$d\\hat{v} = \\hat{w}$', color='red', linestyle='-')
    # plt.plot(bin_edge[:-1] + (bin_edge[1:]-bin_edge[:-1])/2, val)
    # plt.show()
else:
    # in case of the known weighting function we can compare with the true weighting
    if experiment_type == experiment_types[4]:
        z = np.linspace(-10, 10, 500)
        chaining_hat = model.model1(torch.from_numpy(z).float().unsqueeze(1)).squeeze().detach().numpy()
        plt.plot(z, chaining_hat, label='Estimated $\hat{\\nu}$', linestyle='--', color='red')
        plt.legend()
        plt.show()

        w = np.diff(chaining(**c_func_args)(z), 1)
        w_hat = np.diff(chaining_hat, 1)
        plt.plot(z[:-1], w_hat, label='$d\hat{\\nu} = \hat{w}$', color='red', linestyle='--')
        plt.legend()
        plt.show()
    else:    
        z = np.linspace(-10, 10, 500)
        chaining_hat = model.model1(torch.from_numpy(z).float().unsqueeze(1)).squeeze().detach().numpy()
        error_v = np.abs(chaining(**c_func_args)(z) - chaining_hat)
        fig, axes = plt.subplots(2,1,sharex=True, height_ratios=[4,1], figsize=(7,7))
        plt.subplots_adjust(hspace=0.025)
        axes[0].set_title(f'Chaining function {c_func_args if c_func_args["c_type"] != "sumsigmoids" else ""}') if show_plot_titles else None
        axes[0].plot(z, chaining(**c_func_args)(z), label='True $\\nu$', linewidth=4, color='blue')
        axes[0].plot(z, chaining_hat, label='Estimated $\hat{\\nu}$', linestyle='--', color='red')
        # axes[0].plot(z, chaining_hat+error_v.mean(), label='Mean error adjusted $\hat{\\nu}$', linestyle='--', color='magenta')
        axes[0].plot([], [], 'k', label=f'$AE(\\nu,\hat{{\\nu}})$')
        axes[0].hlines([], [], [], 'k', linestyle=':', label=f'$MAE(\\nu,\hat{{\\nu}})={error_v.mean():2.4f}$')
        axes[0].legend()
        axes[0].grid(alpha=0.25)
        axes[0].set_xlabel('$z$')
        axes[0].set_ylabel('$\\nu(z)$')
        axes[1].plot(z, error_v, 'k', label=f'$MAE(\\nu,\hat{{\\nu}})$')
        axes[1].hlines(error_v.mean(), z.min(), z.max(), 'k', linestyle=':', label=f'$\overline{{MAE}}(\\nu,\hat{{\\nu}})={error_v.mean():2.4f}$')
        axes[1].grid(alpha=0.25)
        axes[1].set_xlabel('$z$')
        axes[1].set_ylabel('$Error$')
        # axes[1].legend()
        plt.savefig(f'figs/{save_fig_path}/{save_fig_id}_result_chaining_and_error_ctype_{c_func_args["c_type"]}_{set_str}.pdf', dpi=300, bbox_inches='tight')
        plt.show()

        w = np.diff(chaining(**c_func_args)(z), 1)
        w_hat = np.diff(chaining_hat, 1)
        error_w = np.abs(w - w_hat)
        #
        fig, axes = plt.subplots(2,1,sharex=True, height_ratios=[4,1], figsize=(7,7))
        plt.subplots_adjust(hspace=0.025)
        axes[0].set_title('Derivative of the chaining function (np.diff)') if show_plot_titles else None
        axes[0].plot(z[:-1], w, label='$d\\nu = w$', color='blue')
        axes[0].plot(z[:-1], w_hat, label='$d\hat{\\nu} = \hat{w}$', color='red', linestyle='--')
        axes[0].plot([],[], 'k', label=f'$AE(w,\hat{{w}})$')
        axes[0].hlines([], [], [], 'k', linestyle=':', label=f'$MAE(w,\hat{{w}})={error_w.mean():2.4f}$')
        axes[0].legend()
        axes[0].grid(alpha=0.25)
        axes[0].set_xlabel('$z$')
        axes[0].set_ylabel('$\\nu(z)$')
        #
        axes[1].plot(z[:-1], error_w, 'k', label=f'$MAE(\\nu,\hat{{\\nu}})$')
        axes[1].hlines(error_w.mean(), z[:-1].min(), z[:-1].max(), 'k', linestyle=':', label=f'$\overline{{MAE}}(\\nu,\hat{{\\nu}})={error_w.mean():2.4f}$')
        axes[1].grid(alpha=0.25)
        axes[1].set_xlabel('$z$')
        axes[1].set_ylabel('$Error$')
        # axes[1].legend()
        plt.savefig(f'figs/{save_fig_path}/{save_fig_id}_result_weighting_and_error_ctype_{c_func_args["c_type"]}_{set_str}.pdf', dpi=300, bbox_inches='tight')
        plt.show()

#%%

# from scipy.stats import rv_histogram

# hist = np.histogram(downstream_values_t.flatten(), bins=15)
# hist_dist = rv_histogram(hist, density=False)

# x_ = np.linspace(0, 1000, 1000)
# plt.hist(downstream_values_t,density=True, bins=15)
# plt.plot(x_, hist_dist.pdf(x_))
# plt.show()
# hist_dist.cdf(z_star_2)

# shape=2
# true_scale = 50 # Slightly scaled true Weibull distribution
# true_shift = 10  # Slightly shifted true Weibull distribution
# samples_obs = true_scale * np.random.weibull(a=shape, size=(n_observations,)) + true_shift

# from scipy.stats import weibull_min as weibull
# weibull.cdf(x, a=shape, loc=true_shift, scale=50)
# samples = weibull.rvs(c=shape, loc=true_shift, scale=true_scale, size=n_observations)
# plt.hist(samples_obs)
# plt.hist(samples)
# plt.show()
#%%

def chaining_wrapper(x):
    return model.model1(x)+0

crps1 = crps_ensemble_torch(obs, preds, chaining_wrapper).detach().numpy()
plt.hist(crps_weighted.mean(axis=1), alpha=1.0, hatch='/', edgecolor='k',fill=True)
plt.hist(crps1.mean(axis=1), alpha=0.5)
plt.show()

#%%
crps2 = crps_ensemble_torch(obs[:n_observations//2].unsqueeze(-1), obs[n_observations//2:].unsqueeze(-1), chaining_wrapper).detach().numpy()

plt.hist(crps_weighted.mean(axis=1), alpha=1.0, hatch='/', edgecolor='k',fill=True)
plt.hist(crps1.mean(axis=1), alpha=0.5)
plt.hist(crps2.mean(axis=1), alpha=0.5)
plt.show()