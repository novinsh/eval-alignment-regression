# Overview

This directory self-contain both predicting modeling and downstream decision modeling for the inventory optimization task.

To reproduce the results:
-
- `prep_tsukiji_data.py`: to prep the subdatasets from the original data `tokyo_wholesale_tuna_prices.csv`
- `demand_forecast_dats.py`: to obtain probabilistic predictions based on the subset. The subset has to be chosen manually by setting the correct name from inside of this script.
- `optimization_task.py`: to obtain profits based on the subset. Simiarly, the subset has to be set manually from inside of this file.


Data preparation overview:
-
- convert jpy to eur
- obtain subsets based on certain attributes of the dataset
- some plots and visualization for eda and diagnosis purposes
- save subsets into the `./datasets/` directory


Demand Forecast overview:
-
In the paper we refer to this step as upstream. The dataset contains price and demand values. We are interested in predicting the demand which later will be used as input to the inventory optimization task. Actual price is used as input to the downstream task. 

- some basic time-series analysis based on ACF
- simple probabilitic modeling based on darts library
- predictions are issued for validation and test sets.
- since the data is a time-series, the validation set is created through backtesting over the train range with 1-month forecast horizon and 1-step stride.
- save results (predictions) under the `./predictions/` directory


Inventory Optimization overview:
-
This is the downstream task where the demand predictions are used. 

- Stochastic optimization using pyomo
- save results (profits) under the `./profits/` directory alongside observation and prediction for easier reference later from the single location.
