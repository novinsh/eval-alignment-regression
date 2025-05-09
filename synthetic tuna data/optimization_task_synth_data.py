#%% import packages
solver = 'appsi_highs'
 
import pyomo.environ as pyo
SOLVER = pyo.SolverFactory(solver)
assert SOLVER.available(), f"Solver {solver} is not available."

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


#%% analytical solution

# Setting parameters
c = 10 # cost of purchase
p = 25 # selling price
h = 3  # storage cost

distributions = {
    "Uniform": stats.uniform(loc=25, scale=150),
    "Pareto": stats.pareto(scale=50, b=2),
    "Weibull": stats.weibull_min(scale=112.838, c=2),
    # "Pareto Sharp": stats.pareto(scale=99, b=100),
}

for name, distribution in distributions.items():
    print(f"Mean of {name} distribution = {distribution.mean():0.2f}")

# show PDFs
x = np.linspace(0, 250, 1000)
fig, ax = plt.subplots(1, 1, figsize=(10, 4))
lines = []
for name, distribution in distributions.items():
    ax.plot(x, distribution.pdf(x), lw=3, label=name)
ax.legend()
fig.tight_layout()
plt.show()

# quantile
q = (p - c) / (p + h)

# show CDFs and graphical solutions
extraticks = [q]
fig, ax = plt.subplots(1, 1, figsize=(10, 4))
ax.axhline(q, linestyle="--", color="k")
for name, distribution in distributions.items():
    x_opt = distribution.ppf(q)
    (line,) = ax.plot(x, distribution.cdf(x), lw=3, label=name)
    c = line.get_color()
    ax.plot([x_opt] * 2, [-0.05, q], color=c)
    ax.plot(x_opt, q, ".", color=c, ms=15)
plt.yticks(list(plt.yticks()[0]) + [q], list(plt.yticks()[1]) + [" q "])
plt.ylim(-0.05, 1.1)
ax.legend()
fig.tight_layout()

print(f"The quantile of interest given the parameters is equal to q = {q:.4f}.\n")

for name, distribution in distributions.items():
    x_opt = distribution.ppf(q)
    print(f"The optimal solution for {name} distribution is: {x_opt:0.2f} tons")

#%% Deterministic solution (for average demand)
# finding average optimal decision under different demand distributions.
# this is a test to see how the optimal decision varies under different distributions.

# problem data
c = 10
p = 25
h = 3


def SeafoodStockDeterministic():
    model = pyo.ConcreteModel(
        "Seafood distribution center - Deterministic average demand"
    )

    # key parameter for possible parametric study
    model.mean_demand = pyo.Param(initialize=100, mutable=True)

    # first stage variables and expressions
    model.x = pyo.Var(domain=pyo.NonNegativeReals)

    @model.Expression()
    def first_stage_profit(m):
        return -c * model.x

    # second stage variables, constraints, and expressions
    model.y = pyo.Var(domain=pyo.NonNegativeReals)
    model.z = pyo.Var(domain=pyo.NonNegativeReals)

    @model.Constraint()
    def cant_sell_fish_i_dont_have(m):
        return m.y <= m.mean_demand

    @model.Constraint()
    def fish_do_not_disappear(m):
        return m.y + m.z == m.x

    @model.Expression()
    def second_stage_profit(m):
        return p * m.y - h * m.z

    # objective
    @model.Objective(sense=pyo.maximize)
    def total_profit(m):
        return m.first_stage_profit + m.second_stage_profit

    return model


model = SeafoodStockDeterministic()
result = SOLVER.solve(model)
assert result.solver.status == "ok"
assert result.solver.termination_condition == "optimal"

print(
    f"Optimal solution for determistic demand equal to the average demand = {model.x():.1f} tons"
)
print(f"Optimal deterministic profit = {model.total_profit():.0f}€")

#%% naive sample average approximation 
# fixing the optimization variable to 100 and observing the price
# this is a test to see how the profit changes under fixed sub-optimal decision.

def NaiveSeafoodStockSAA(N, sample, distributiontype):
    model = pyo.ConcreteModel(
        "Seafood distribution center - Naive solution performance"
    )

    def indices_rule(model):
        return range(N)

    model.indices = pyo.Set(initialize=indices_rule)
    model.xi = pyo.Param(model.indices, initialize=dict(enumerate(sample)))

    # first stage variable: x (amount of fish bought)
    model.x = 100.0

    def first_stage_profit(model):
        return -c * model.x

    model.first_stage_profit = pyo.Expression(rule=first_stage_profit)

    # second stage variables: y (sold) and z (unsold)
    model.y = pyo.Var(model.indices, domain=pyo.NonNegativeReals)
    model.z = pyo.Var(model.indices, domain=pyo.NonNegativeReals)

    # second stage constraints
    model.cantsoldthingsfishdonthave = pyo.ConstraintList()
    model.fishdonotdisappear = pyo.ConstraintList()
    for i in model.indices:
        model.cantsoldthingsfishdonthave.add(expr=model.y[i] <= model.xi[i])
        model.fishdonotdisappear.add(expr=model.y[i] + model.z[i] == model.x)

    def second_stage_profit(model):
        return sum([p * model.y[i] - h * model.z[i] for i in model.indices]) / float(N)

    model.second_stage_profit = pyo.Expression(rule=second_stage_profit)

    def total_profit(model):
        return model.first_stage_profit + model.second_stage_profit

    model.total_expected_profit = pyo.Objective(rule=total_profit, sense=pyo.maximize)

    result = SOLVER.solve(model)

    print(
        f"- {distributiontype}-distributed demand: {model.total_expected_profit():.2f}€ - optimal x: {model.x}"
    )

    return model.total_expected_profit()


np.random.seed(0)
N = 5000
print(
    f"Approximate expected optimal profit calculate with {N} samples when assuming the average demand with"
)

samples = np.random.uniform(low=25.0, high=175.0, size=N)
naiveprofit_uniform = NaiveSeafoodStockSAA(N, samples, "Uniform")

shape = 2
xm = 50
samples = (np.random.pareto(a=shape, size=N) + 1) * xm
naiveprofit_pareto = NaiveSeafoodStockSAA(N, samples, "Pareto")

shape = 2
scale = 113
samples = scale * np.random.weibull(a=shape, size=N)
naiveprofit_weibull = NaiveSeafoodStockSAA(N, samples, "Weibull")

samples = np.random.uniform(low=0, high=200, size=N)
naiveprofit_uniform = NaiveSeafoodStockSAA(N, samples, "Uniform Wide")

shape = 100
xm = 99
samples = (np.random.pareto(a=shape, size=N) + 1) * xm
naiveprofit_pareto = NaiveSeafoodStockSAA(N, samples, "Pareto Sharp")



#%% sample average approximation solution
# finding optimal decision under different distributions.
# this is to find the optimal decision as opposed to the previous solution which was fixed
# by comparing the optimal solution here with the naive solution, we can see the gain
# by finding the optimal solution.

def SeafoodStockSAA(N, sample):
    model = pyo.ConcreteModel("Seafood Stock using SAA method")

    def indices_rule(model):
        return range(N)

    model.indices = pyo.Set(initialize=indices_rule)
    model.xi = pyo.Param(model.indices, initialize=dict(enumerate(sample)))

    # first stage variable: x (amount of fish bought)
    model.x = pyo.Var(domain=pyo.NonNegativeReals)

    def first_stage_profit(model):
        return -c * model.x

    model.first_stage_profit = pyo.Expression(rule=first_stage_profit)

    # second stage variables: y (sold) and z (unsold)
    model.y = pyo.Var(model.indices, domain=pyo.NonNegativeReals)  # sold
    model.z = pyo.Var(
        model.indices, domain=pyo.NonNegativeReals
    )  # unsold to be returned

    # second stage constraints
    model.cantsoldfishidonthave = pyo.ConstraintList()
    model.fishdonotdisappear = pyo.ConstraintList()
    for i in model.indices:
        model.cantsoldfishidonthave.add(expr=model.y[i] <= model.xi[i])
        model.fishdonotdisappear.add(expr=model.y[i] + model.z[i] == model.x)

    def second_stage_profit(model):
        return sum([p * model.y[i] - h * model.z[i] for i in model.indices]) / float(N)

    model.second_stage_profit = pyo.Expression(rule=second_stage_profit)

    def total_profit(model):
        return model.first_stage_profit + model.second_stage_profit

    model.total_expected_profit = pyo.Objective(rule=total_profit, sense=pyo.maximize)

    model.profit_per_sample = pyo.Expression(
        range(N),
        rule=lambda m, i: p * m.y[i] - c * m.x - h * m.z[i]
    )

    return model


def display_results(model, distributiontype):
    smartprofit = model.total_expected_profit()
    print(
        f"Approximate solution in the case of {distributiontype} distribution using N={N:.0f} samples"
    )
    print(f"Approximate optimal solution: x = {model.x.value:.2f} tons")
    print(f"Approximate expected optimal profit: {smartprofit:.2f}€")
    if distributiontype == "uniform":
        print(
            f"Value of the stochastic solution: {smartprofit:.2f}-{naiveprofit_uniform:.2f} = {smartprofit-naiveprofit_uniform:.2f}€\n"
        )
    elif distributiontype == "Pareto":
        print(
            f"Value of the stochastic solution: {smartprofit:.2f}-{naiveprofit_pareto:.2f} = {smartprofit-naiveprofit_pareto:.2f}€\n"
        )
    elif distributiontype == "Weibull":
        print(
            f"Value of the stochastic solution: {smartprofit:.2f}-{naiveprofit_weibull:.2f} = {smartprofit-naiveprofit_weibull:.2f}€\n"
        )
    return None


np.random.seed(1)
N = 5000

def plot_profit_dist(profits_per_sample):
    plt.hist(profits_per_sample)
    plt.vlines(np.mean(profits_per_sample), 0, 1000, color='red', label='mean')
    plt.legend()
    plt.show()

get_profits_per_sample = lambda model, N: [pyo.value(model.profit_per_sample[i]) for i in range(N)]

samples = np.random.uniform(low=25.0, high=175.0, size=N)
model = SeafoodStockSAA(N, samples)
SOLVER.solve(model)
display_results(model, "uniform")
profits_per_sample = get_profits_per_sample(model, N)
plot_profit_dist(profits_per_sample)


shape = 2
xm = 50
samples = (np.random.pareto(a=shape, size=N) + 1) * xm
model = SeafoodStockSAA(N, samples)
SOLVER.solve(model)
display_results(model, "Pareto")
profits_per_sample = get_profits_per_sample(model, N)
plot_profit_dist(profits_per_sample)


shape = 2
scale = 113
samples = scale * np.random.weibull(a=shape, size=N)
model = SeafoodStockSAA(N, samples)
SOLVER.solve(model)
display_results(model, "Weibull")
profits_per_sample = get_profits_per_sample(model, N)
plot_profit_dist(profits_per_sample)

#%%
# a diagnosis to see all variables affected by the demand sample.
# the decision variable x is fixed in here while demand varaible y varies.

optimal_solutions = []
for i in range(N):
    x_opt = pyo.value(model.x)  # First-stage decision (shared)
    y_opt = pyo.value(model.y[i])  # Second-stage decision (sales)
    z_opt = pyo.value(model.z[i])  # Second-stage decision (leftover inventory)
    optimal_solutions.append((x_opt, y_opt, z_opt))

# Print the optimal solutions for each sample
for i, (x_opt, y_opt, z_opt) in enumerate(optimal_solutions):
    print(f"Sample {i + 1}: x = {x_opt:.2f}, y = {y_opt:.2f}, z = {z_opt:.2f}")

#%% run the experiment to save the results for the alignment purposes
# only evaluate based on predictive distribution samples

n_observations = 1000
n_scenarios = 100 
shape = 2
scale = 113
samples_pred = scale * np.random.weibull(a=shape, size=(n_observations, n_scenarios))
samples_obs = scale * np.random.weibull(a=shape, size=(n_observations,))

print(samples_pred.shape)
print(samples_obs.shape)
#%% run the optimization

profits = []
expected_profits = [] # for diagnosis
for i in range(n_observations):
    model = SeafoodStockSAA(n_scenarios, samples_pred[i])
    SOLVER.solve(model)
    # display_results(model, "Weibull")
    profits_per_sample = get_profits_per_sample(model, n_scenarios)
    if i % 100 == 0:
        print(f"{i} - optimal x: {model.x.value}, profit: {model.total_expected_profit()}, profit2: {np.mean(profits_per_sample)}")
    profits.append(profits_per_sample)
    expected_profits.append(model.total_expected_profit())
    # plot_profit_dist(profits_per_sample)
profits = np.array(profits)
expected_profits = np.array(expected_profits)

assert np.isclose(np.mean(profits, axis=1), expected_profits, rtol=1e-3, atol=1e-5).all(), "something wrong with the expected profit and per sample profit calculation"
#%% save the results to the file
# samples_pred.shape
# samples_obs.shape
# profits.shape
# np.save("downstream_data/seafood_weibull_preds.npy", samples_pred)
# np.save("downstream_data/seafood_weibull_obs.npy", samples_obs)
# np.save("downstream_data/seafood_weibull_profits.npy", profits)

#%% optimize based on predictive distribution and evaluate based on ground truth distribution

class SeafoodStockSAAClass:
    """ modification of the original class to include methods for evaluation and ease of use """
    def __init__(self, N, sample):
        self.N = N
        self.sample = sample
        self.model = self._build_model()

    def _build_model(self):
        model = pyo.ConcreteModel("Seafood Stock using SAA method")

        def indices_rule(model):
            return range(self.N)

        model.indices = pyo.Set(initialize=indices_rule)
        model.xi = pyo.Param(model.indices, initialize=dict(enumerate(self.sample)))

        # first stage variable: x (amount of fish bought)
        model.x = pyo.Var(domain=pyo.NonNegativeReals)

        def first_stage_profit(model):
            return -c * model.x

        model.first_stage_profit = pyo.Expression(rule=first_stage_profit)

        # second stage variables: y (sold) and z (unsold)
        model.y = pyo.Var(model.indices, domain=pyo.NonNegativeReals)  # sold
        model.z = pyo.Var(model.indices, domain=pyo.NonNegativeReals)  # unsold

        # second stage constraints
        model.cantsoldfishidonthave = pyo.ConstraintList()
        model.fishdonotdisappear = pyo.ConstraintList()
        for i in model.indices:
            model.cantsoldfishidonthave.add(expr=model.y[i] <= model.xi[i])
            model.fishdonotdisappear.add(expr=model.y[i] + model.z[i] == model.x)

        def second_stage_profit(model):
            return sum([p * model.y[i] - h * model.z[i] for i in model.indices]) / float(self.N)

        model.second_stage_profit = pyo.Expression(rule=second_stage_profit)

        def total_profit(model):
            return model.first_stage_profit + model.second_stage_profit

        model.total_expected_profit = pyo.Objective(rule=total_profit, sense=pyo.maximize)

        return model

    def solve(self, solver):
        solver.solve(self.model)

    def get_total_expected_profit(self):
        return pyo.value(self.model.total_expected_profit)

    def get_x(self):
        return pyo.value(self.model.x)

    def evaluate(self, samples):
        profit_per_sample = [
            # equivalent of p * m.y[i] - c * self.model.x.value - h * self.model.z[i]:
            # selling price * amount sold - buying price * amount bought - holding price * unsold/remained amount
            # amoung sold: maximum is the amount that is bought/available
            # unsold/remained amount: amount bought/available - demand
            p * min(samples[i], self.model.x.value) - c * self.model.x.value - h * max(0, self.model.x.value - samples[i])
            for i in range(len(samples))
        ]
        return np.array(profit_per_sample)


# code without using the class and using the previously defined functions
# this was to test the class implementation is correct.
# shape = 2
# # xm = 50
# scale = 113
# true_scale = 113 # Slightly scaled true Weibull distribution
# true_shift = 0   # Slightly shifted true Weibull distribution
# N = 1000

# np.random.seed(0)
# # Predictive Weibull distribution (used for optimization)
# predictive_samples = scale * np.random.weibull(a=shape, size=N)
# model = SeafoodStockSAA(N, predictive_samples)
# SOLVER.solve(model)
# # model.solve(SOLVER)
# expected_profit, xw = model.total_expected_profit(), model.x.value
# profits_per_sample = get_profits_per_sample(model, N)
# print(expected_profit)
# print(np.mean(profits_per_sample))
# print(xw)

# sample use case of the class 1 predictive sample per observation.
shape = 2
# xm = 50
scale = 113
true_scale = 113 # Slightly scaled true Weibull distribution
true_shift = 0  # Slightly shifted true Weibull distribution
N = 1000

np.random.seed(0)
# Predictive Weibull distribution (used for optimization)
predictive_samples = scale * np.random.weibull(a=shape, size=N)
model = SeafoodStockSAAClass(N, predictive_samples)
model.solve(SOLVER)
expected_profit, xw = model.get_total_expected_profit(), model.get_x()
profits_per_sample = model.evaluate(predictive_samples)
print("expected profit on predictive distribution: ", expected_profit)
print("average expected profit: ", np.mean(profits_per_sample))
print("optimal decision: ", xw)

# True Weibull distribution (used for evaluation)
true_samples = true_scale * np.random.weibull(a=shape, size=N) + true_shift
evaluated_profits_per_sample = model.evaluate(true_samples)
expected_evaluated_profit = evaluated_profits_per_sample.mean()
# evaluated_profit_per_sample = np.array([min(d, xw) * (100 - d) for d in true_samples])
print("evaluated expected profit on test set: ", expected_evaluated_profit)
print("average expected profit on test set: ", np.mean(evaluated_profits_per_sample))
print("(fixed) optimal decision under test: ", xw)

plt.hist(predictive_samples, label='predictive distribution', density=True)
plt.hist(true_samples, label='true distribution', density=True, alpha=0.5)
plt.legend()
plt.show()

plt.hist(profits_per_sample, label='expected profit (based on predictive distribution)')
plt.hist(evaluated_profits_per_sample, label='realized profit (based on observation)', alpha=0.5)
plt.vlines(expected_profit, 0, 100, color='blue', label='average expected profit')
plt.vlines(expected_evaluated_profit, 0, 100, color='orange', label='average realized profit')
plt.legend()
plt.show()

#%% run the experiment to save the results for the alignment purposes
# evaluate based on predictive distribution samples as well as observation samples

# create data where predictive distribution has n_scenarios per observation
n_observations = 1000
n_scenarios = 100 
optimal_prediction = False # set to True for the case predictive distribution is same as test distribution, otherwise set to False
shape = 2
true_scale = 50 # Slightly scaled true Weibull distribution
true_shift = 20  # Slightly shifted true Weibull distribution
if optimal_prediction:
    scale = true_scale
    shift = true_shift
else:
    scale = 100#113
    shift = 20
np.random.seed(0)
samples_pred = scale * np.random.weibull(a=shape, size=(n_observations, n_scenarios)) + shift
samples_obs = true_scale * np.random.weibull(a=shape, size=(n_observations,)) + true_shift

print(samples_pred.shape)
print(samples_obs.shape)

plt.hist(samples_obs, label='true distribution', density=True,)
plt.hist(samples_pred.flatten(), label='predictive distribution', density=True, alpha=0.5)
plt.legend()
plt.show()

# stats.weibull_min(scale=true_scale, c=shape, loc=true_shift).ppf((p-c)/(p+h))

#%% run the optimization pipeline per each test observation
# optimize based on the predictive distribution and evaluate for each test sample (observed)

profits_pred = [] # profits calculated based on predictive samples
expected_profits_pred = [] # for diagnosis
profits_obs = [] # profits calcualted based on observation samples

for i in range(n_observations):
    model = SeafoodStockSAAClass(n_scenarios, samples_pred[i])
    model.solve(SOLVER)
    expected_profit, xw = model.get_total_expected_profit(), model.get_x()
    profits_per_sample = model.evaluate(samples_pred[i])
    if i % 100 == 0:
        print(f"{i} - optimal x: {model.get_x()}, profit: {model.get_total_expected_profit()}, profit2: {np.mean(profits_per_sample)}")
    profits_pred.append(profits_per_sample)
    expected_profits_pred.append(model.get_total_expected_profit())
    profits_obs.append(model.evaluate(samples_obs[[i]]))
    # plot_profit_dist(profits_per_sample)
profits_pred = np.array(profits_pred)
expected_profits_pred = np.array(expected_profits_pred)
profits_obs = np.array(profits_obs)
expected_profits_obs = np.mean(profits_obs)

assert np.isclose(np.mean(profits_pred, axis=1), expected_profits_pred, rtol=1e-3, atol=1e-5).all(), "something wrong with the expected profit and per sample profit calculation"

#%% visualization
# plt.hist(profits_pred.mean(axis=1))
# plt.vlines(profits_obs.mean(), 0, 100)
# plt.show()
plt.hist(profits_pred.flatten(), label='profit based on predictive samples', density=True)
plt.hist(profits_obs.flatten(), label='profit based on observations (realized)', alpha=0.5, density=True)
_,_,_,ymax = plt.axis()
plt.vlines(expected_profits_pred.mean(), 0, ymax/4, color='blue', label='average expected profit')
plt.vlines(expected_profits_obs, 0, ymax/4, color='orange', linestyle='--', label='average realized profit')
# plt.hist(profits_pred.mean(axis=1), label='expected profit based on predictive samples', color='blue', alpha=0.5, density=True)
# plt.ylim(0,0.001)
plt.legend()
plt.show()
#%% save the results to the file
# samples_pred.shape
# samples_obs.shape
# profits.shape

optimal_str = 'optimal' if optimal_prediction else 'nonoptimal'

np.save(f"datasets/seafood_weibull_{optimal_str}_preds.npy", samples_pred)
np.save(f"datasets/seafood_weibull_{optimal_str}_obs.npy", samples_obs)
np.save(f"datasets/seafood_weibull_{optimal_str}_profits_pred.npy", profits_pred)
np.save(f"datasets/seafood_weibull_{optimal_str}_profits_obs.npy", profits_obs)

#%%
print(samples_pred.shape)
print(samples_obs.shape)
print(profits_pred.shape)
print(profits_obs.shape)