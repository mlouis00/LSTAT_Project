
#### Import
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import statsmodels.api as sm

from tqdm import tnrange, tqdm_notebook
import itertools
import time

from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn import metrics
from sklearn import datasets
import sklearn

import preprocessing as pre
import tools

def test_type2(x,y):
    #x, y = np.array(x), np.array(y)
    #drop_list = [12,8,20,15,9,11,18]
    x_update = x
    drop_list = ["GraduationYear","Specialization","extraversion","ComputerProgramming","CollegeCityTier", "Logical", "openess_to_experience"]
    x_update = sm.add_constant(x_update)
    for i in drop_list:
        #x_update = np.delete(x_update, i, axis=1)
        #x_update.reset_index(drop=True, inplace=True)
        x_update = x_update.drop(i, axis=1)
    model = sm.OLS(endog = y, exog = x_update)
    results = model.fit()
    print(results.summary())
    '''
    '''
    print('coefficient of determination:', results.rsquared)

    print('adjusted coefficient of determination:', results.rsquared_adj)

    print('regression coefficients:', results.params)
    print('regression coefficients:', results)