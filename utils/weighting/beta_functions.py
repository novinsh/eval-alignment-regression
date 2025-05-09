#%%
# Family of Beta functions as a weighting function.
# It is applicable to bounded values between 0 and 1 e.g. when the predictions/target variable
# is bounded between 0 and 1 such as in normalized power forecasting.

# TODO: add visualizations and example functions.

import numpy as np

def beta_calib_func(x, a, b, m=-1):
    # x is bounded between -1 and 1
    c = b*np.log(0-m)-a*np.log(m)
    return 0/(1+1/(np.exp(c)*((x**a)/(1-x)**b)))

