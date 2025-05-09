#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import NaiveSeasonal, ExponentialSmoothing, NBEATSModel, LinearRegressionModel
from darts.metrics import mape, smape
from darts.dataprocessing.transformers import Scaler
from darts.utils.statistics import check_seasonality, plot_acf, plot_hist
from darts.utils.model_selection import train_test_split

dataset_names = [
                "Bluefin_Fresh_JapaneseFleet",
                "Bluefin_Fresh_ForeignFleet",
                "Bluefin_Frozen_UnknownFleet",
                "Southern_Fresh_UnknownFleet",
                "Southern_Frozen_UnknownFleet",
                "Bigeye_Fresh_UnknownFleet"
]

# TODO: script it to iterate over all datasets 
dataset_name = dataset_names[3]

# Load and prepare data
df = pd.read_csv(f"datasets/tuna_{dataset_name}.csv", index_col="date", parse_dates=True, date_parser=pd.to_datetime,)
series = TimeSeries.from_dataframe(df, value_cols='demand')

# Split into train/test 
# We don't define a validation set explicitely but use the backtesting API that
# Produces forecasts in an expanding window fashion in a way that emulates the 
# validation data that we need
train_, test_ = train_test_split(series, test_size=24)

# Normalize the series
scaler = Scaler()
scaler.fit(train_)
train = scaler.transform(train_)
test =scaler.transform(test_)

print(train.shape)
print(test.shape)

plt.figure(figsize=(12, 6))
train.plot(label="train")
test.plot(label="test")
plt.savefig(f"../figs/tsukiji_seafood/{dataset_name}_train_test_subset2.pdf", bbox_inches='tight', dpi=300)
plt.show()

# Seasonality analysis with ACF
plot_acf(train, m=12, alpha=0.05, max_lag=24) 

for m in range(2, 25):
    is_seasonal, period = check_seasonality(train, m=m, alpha=0.05)
    if is_seasonal:
        print(f"There is seasonality of order {period}.")

#%% Create the model
# Fit a naive seasonal model with seasonal period K=12 (monthly data)
probabilistic_model = True
single_shot_prediction = False
if probabilistic_model:
    # model = ExponentialSmoothing(trend=ModelMode.ADDITIVE, seasonal=SeasonalityMode.ADDITIVE, seasonal_periods=12, damped=True)
    model = ExponentialSmoothing(seasonal_periods=12, damped=True)
    # model = LinearRegressionModel(
    #     lags=12,
    #     # lags_future_covariates=[0],
    #     likelihood="quantile",
    #     quantiles=[0.05, 0.1, 0.5, 0.9, 0.95],
    # ) 
else:
    model = NaiveSeasonal(K=12)
    # model = NBEATSModel(input_chunk_length=12, output_chunk_length=1)
    # model = AutoARIMA()

#  fitting procedure 
if single_shot_prediction:
    model.fit(train)
    # Forecast the test set length
    if probabilistic_model:
        num_samples = 100
        # forecast = model.predict(len(val), num_samples=num_samples)
        # forecast = model.predict(len(val)+len(test), num_samples=num_samples)
        forecasts_te = model.predict(len(test), num_samples=num_samples)
    else:
        num_samples = 1
        # forecast = model.predict(len(val), num_samples=num_samples)
        forecasts_te = model.predict(len(test), num_samples=num_samples)
else: # fitting procedure with expanding window prediction
    # List to store predictions
    predictions = []
    # train_tmp = copy.deepcopy(train)
    train_tmp = train.copy()
    # Expanding window retraining
    for i in range(len(test)):
        # Train the model on the expanding window (train + first i test points)
        model.fit(train_tmp)
        # Predict the next value
        num_samples = 100 if probabilistic_model else 1
        prediction = model.predict(1, num_samples=num_samples)
        predictions.append(prediction)
        # Optionally, add the predicted value to the training data
        # or use actual test values for retraining
        train_tmp = train_tmp.append(test[i])
    # Convert predictions to a TimeSeries object
    timestamps = [p.time_index[0] for p in predictions]
    values = np.concatenate([p.all_values() for p in predictions], axis=0)

    forecasts_te = TimeSeries.from_times_and_values(
        times=pd.DatetimeIndex(timestamps),
        values=values
    )

# Inverse transform to original scale
forecasts_te = scaler.inverse_transform(forecasts_te).map(lambda x: np.clip(x, a_min=0, a_max=None))

# Plot the full historical series and forecast
plt.figure(figsize=(12, 6))
train_.plot(label="train")
test_.plot(label="test")
forecasts_te.plot(label="Forecast")
plt.axvline(x=test.start_time(), color='gray', linestyle='--', label="Test Start")
plt.legend()
plt.grid(True)
plt.show()

# Compute and print MAPE
print(f"MAPE on test set: {mape(test_, forecasts_te):.2f}%")
print(f"SMAPE on test set: {smape(test_, forecasts_te):.2f}%")

fig, ax = plt.subplots()
plot_hist(test_, ax=ax, density=True)
plot_hist(forecasts_te, ax=ax, density=True)
ax.legend(['Test', 'Forecast'])
plt.show()
#%% backtesting - 1-month ahead forecast expanding window from beginning of train and last point in each window as validation
historical_forecasts = model.historical_forecasts(train, forecast_horizon=1, retrain=True, stride=1, last_points_only=True, num_samples=num_samples)
forecasts_val = scaler.inverse_transform(historical_forecasts).map(lambda x: np.clip(x, a_min=0, a_max=None))
train_.plot(label="historical observations")
forecasts_val.plot(label="1-month ahead forecast")
print(f"1-month historical forecast expanding window MAPE: {mape(train_, forecasts_val):2.2f}%")

#%%
fig, ax = plt.subplots()
plot_hist(train_, ax=ax, density=True)
plot_hist(forecasts_val, ax=ax, density=True)
ax.legend(['Historical observations', 'Forecasts'])
plt.show()

#%%
# Save the predictions
forecasts_val_ = forecasts_val.to_dataframe()
forecasts_val_.columns = [i for i in range(num_samples)]
forecasts_val_.to_csv(f"predictions/tuna_{dataset_name}_preds_val.csv", index_label="date", float_format="%.2f")
#
forecasts_te_ = forecasts_te.to_dataframe()
forecasts_te_.columns = [i for i in range(num_samples)]
forecasts_te_.to_csv(f"predictions/tuna_{dataset_name}_preds_te.csv", index_label="date", float_format="%.2f")

# no need to save observations since it's already in the dataset under demand column!
# pd.Series(obs_numpy, index=forecasts.time_index).to_csv(f"tuna_obs_{dataset_name}.csv")
# np.save(f"tuna_obs_{dataset_name}.npy", obs_numpy)


#%% 12-month ahead forecast with 12 month stride (for test) - just for testing
historical_forecasts = model.historical_forecasts(train, forecast_horizon=12, stride=12, last_points_only=False, num_samples=num_samples)

train_.plot(label="train", linestyle='--')
for h in historical_forecasts:
    scaler.inverse_transform(h).map(lambda x: np.clip(x, a_min=0, a_max=None)).plot(label="12-month ahead forecast")

# the following error calculation only works if horizon=stride. It requires a contigues series 
forecasts_all = scaler.inverse_transform(concatenate(historical_forecasts)).map(lambda x: np.clip(x, a_min=0, a_max=None))
print(f"12-month historical forecast expanding window MAPE : {mape(train_, forecasts_all):2.2f}%")

