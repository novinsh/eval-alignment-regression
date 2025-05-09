#%% import packages
solver = 'appsi_highs'
 
import pyomo.environ as pyo
SOLVER = pyo.SolverFactory(solver)
assert SOLVER.available(), f"Solver {solver} is not available."

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

#%% optimize based on predictive distribution and evaluate based on ground truth distribution
class SeafoodStockSAAClass:
    """ modification of the original class to include methods for evaluation and ease of use """
    def __init__(self, N, sample, c=10, p=25, h=3):
        self.N = N
        self.sample = sample
        self.c = c 
        self.p = p
        self.h = h
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
            return -self.c * model.x

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
            return sum([self.p * model.y[i] - self.h * model.z[i] for i in model.indices]) / float(self.N)

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
            self.p * min(samples[i], self.model.x.value) - self.c * self.model.x.value - self.h * max(0, self.model.x.value - samples[i])
            for i in range(len(samples))
        ]
        return np.array(profit_per_sample)


#%%
def run_optimization(df, preds, c, p, h):
    n_observations = len(df)
    n_scenarios = preds.shape[1]
    samples_pred = preds
    samples_obs = df['demand'].values

    profits_pred = [] # profits calculated based on predictive samples
    expected_profits_pred = [] # for diagnosis
    profits_obs = [] # profits calcualted based on observation samples

    print(n_observations)
    print(n_scenarios)

    q_vals = []
    critical_qs = []
    for i in range(n_observations):
        # to compare with analytical results save critical q and its corresponding value
        critical_q = (p[i]-c[i])/(p[i]+h)
        q_val = np.quantile(samples_pred[i], critical_q)
        q_vals.append(q_val)
        critical_qs.append(critical_q)
        #
        model = SeafoodStockSAAClass(n_scenarios, samples_pred[i], c=c[i], p=p[i], h=h)
        model.solve(SOLVER)
        expected_profit, xw = model.get_total_expected_profit(), model.get_x()
        profits_per_sample = model.evaluate(samples_pred[i])
        if i % 10 == 0:
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

    # visualization of the profits distribution
    # plt.hist(profits_pred.mean(axis=1))
    # plt.vlines(profits_obs.mean(), 0, 100)
    # plt.show()
    plt.hist(profits_pred.flatten(), label='profit based on predictive samples', density=True)
    plt.hist(profits_obs.flatten(), label='profit based on observations (realized)', alpha=0.5, density=True)
    _,_,_,ymax = plt.axis()
    plt.vlines(expected_profits_pred.mean(), 0, ymax/4, color='blue', label=f'average expected profit ({expected_profits_pred.mean()/1e6:2.2f} Million €)')
    plt.vlines(expected_profits_obs, 0, ymax/4, color='orange', linestyle='--', label=f'average realized profit ({expected_profits_obs/1e6:2.2f} Million €)')
    # plt.hist(profits_pred.mean(axis=1), label='expected profit based on predictive samples', color='blue', alpha=0.5, density=True)
    # plt.ylim(0,0.001)
    plt.xlabel("EUR")
    plt.legend()
    plt.show()

    # visualizetion of the critical quantile values distribution
    plt.title('Critical Quantile Values PDF')
    plt.hist(q_vals)
    plt.xlabel('Tuna demand (tonne)')
    plt.show()

    plt.title('Critical Quantile Values PDF')
    plt.hist(critical_qs)
    plt.show()

    plt.title('Critical Quantile Values PDF')
    val, bin_edge = np.histogram(q_vals, bins=20)
    plt.plot(bin_edge[:-1] + (bin_edge[1:]-bin_edge[:-1])/2, val)
    plt.title('Tuna demand (tonne)')
    plt.show()

    q_vals_sorted = np.sort(q_vals)
    p = 1. * np.arange(len(q_vals)) / (len(q_vals)-1)

    plt.title('Critical Quantile Values CDF')
    plt.plot(q_vals_sorted, p)
    plt.title('Tuna demand (tonne)')
    plt.show()

    # save the results to the file
    # samples_pred.shape
    # samples_obs.shape
    # profits.shape
    return samples_pred, q_vals, samples_obs, profits_pred, profits_obs

#%% load tsukiji data
dataset_names = [
                "Bluefin_Fresh_JapaneseFleet",
                "Bluefin_Fresh_ForeignFleet",
                "Bluefin_Frozen_UnknownFleet",
                "Southern_Fresh_UnknownFleet",
                "Southern_Frozen_UnknownFleet",
                "Bigeye_Fresh_UnknownFleet"
]

# TODO: script it to iterate over parameters. 
# Manually set the parameters to reproduce experiments in the paper
dataset_name = dataset_names[3] # choose dataset
test_set = False        # choose between test and validation set
constant_price = False  # choose between constant or variable price from dataset

df = pd.read_csv(f"datasets/tuna_{dataset_name}.csv", index_col="date", parse_dates=True, date_format='%Y-%m')
df['price'] = df['price'] * 1000 # price per kilo --> price per tonne
# df['demand'] = df['demand'] * 1000 # demand per tonne --> demand per kilo 
if test_set:
    df_pred = pd.read_csv(f"predictions/tuna_{dataset_name}_preds_te.csv", index_col="date", parse_dates=True, date_format='%Y-%m')
else:
    df_pred = pd.read_csv(f"predictions/tuna_{dataset_name}_preds_val.csv", index_col="date", parse_dates=True, date_format='%Y-%m')
df = df.loc[df_pred.index] # select only the months that are in the prediction set
preds = df_pred.values

# make sure the dates are consequative (months)
assert ((pd.to_datetime(df.index).diff()[1:] >= pd.Timedelta(28, unit="d")) & \
            (pd.to_datetime(df.index).diff()[1:] <= pd.Timedelta(31, unit="d"))).all()

print(df['price'].describe())
# Setting parameters
if constant_price:
    # use constant price data
    # c = [10]*len(df)
    # p = [25]*len(df)
    # h = 3
    # average price from the dataset
    c = [df['price'].mean()]*len(df)
    p = [df['price'].mean()*2.5]*len(df)
    h = df['price'].mean()/4
else:
    # use price data from the dataset
    c = df['price'].values # cost of purchase
    p = df['price'].values*2.5 # selling price
    h = 7*1000  # storage cost - constant


# Visualization of Demand, Price and Demand Forecast
fig, ax = plt.subplots(figsize=(14,5))
df['demand'].plot(ax=ax)
# plt.plot(preds, alpha=0.1, color='r')
plt.fill_between(df.index, np.quantile(preds, 0.05, axis=1), np.quantile(preds, 0.95, axis=1), color='red', alpha=0.25, label='90% PI')
plt.fill_between(df.index, np.quantile(preds, 0.2, axis=1), np.quantile(preds, 0.7, axis=1), color='red', alpha=0.45, label='50% PI')
plt.plot(df.index, np.quantile(preds, 0.5, axis=1), color='red', label='median pred')
plt.grid(alpha=0.25)
ax.legend(loc=2)
ax.set_ylabel('Demand (tonne)')
ax2 = ax.twinx()
df['price'].plot(ax=ax2, color='tab:green')
ax2.tick_params(axis='y', labelcolor='tab:green')
ax2.set_ylabel('Price per tonne (€)', color='tab:green')
plt.legend(loc=1)
fig.tight_layout()
plt.savefig(f"../figs/tsukiji_seafood/{dataset_name}_demand_price_predictioHey,n.pdf", dpi=300, bbox_inches='tight')
plt.show()

# run optimization
samples_pred, q_vals, samples_obs, profits_pred, profits_obs = run_optimization(df, preds, c, p, h)

# save the results to the file
save_dir_name = "test" if test_set else "validation"
constant_price_str = "const" if constant_price else "vary"
np.save(f"profits/{save_dir_name}/tuna_{dataset_name}_price-{constant_price_str}_preds.npy", samples_pred)
np.save(f"profits/{save_dir_name}/tuna_{dataset_name}_price-{constant_price_str}_critical_qs.npy", q_vals)
np.save(f"profits/{save_dir_name}/tuna_{dataset_name}_price-{constant_price_str}_obs.npy", samples_obs)
np.save(f"profits/{save_dir_name}/tuna_{dataset_name}_price-{constant_price_str}_profits_pred.npy", profits_pred)
np.save(f"profits/{save_dir_name}/tuna_{dataset_name}_price-{constant_price_str}_profits_obs.npy", profits_obs)