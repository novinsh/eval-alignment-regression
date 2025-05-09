This directory hosts output of the decision-making optimization step provided by `optimization_task.py` based on predictions and observation for each of the dataset subsets. Results are separately provided for the validation and test sets.


**Objects saved**:
-
- observations
- predictions 
- downstream value (profits) per observation
- downstream value (profits) per prediction
- critical quantile value per prediction - the ratio of prices/costs that inform the quantile value of the predictive distribution that matters (analytical result from newsvendor model under certain assumptions)


**Variations of the experiment**:
-
- hypothetical constant prices/costs
- user variable prices/costs available in the dataset

**remarks**:
-
- the predictions and observations are duplicate of what already exists under the `./predictions/` and `./datasets`, however instead of `.csv` format all results are saved here as `.npy` as a matter of convenience to load it later into the alignment procedure from a single source/directory.