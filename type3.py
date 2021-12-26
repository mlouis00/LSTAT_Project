
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

def test_lasso(x, y):
    model_lasso = linear_model.Lasso()
    model_lasso.fit(x, y)
    print(model_lasso)
    print(model_lasso.coef_)
    print(model_lasso.feature_names_in_)
    print(model_lasso.intercept_)