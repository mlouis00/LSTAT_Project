
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
#import type1 as t1
import type2 as t2
import type3 as t3

###Hypothesis:
# The more people do skill tests, and the best their score on those tests and during their studies, the highest their salary.
# We will try to demonstrate (if their exist,) the correlation between school and ACMAT scores and salary.

### init
data = pd.read_csv("Engineering_graduate_salary.csv", sep=",")

length_data = len(data)
subset = round(length_data*.1)

data = pre.drop_useless(data, ["ID", "10board", "12board", "CollegeID", "CollegeCityID", "MechanicalEngg", "ElectricalEngg", "TelecomEngg", "CivilEngg"])
data = pre.date_update(data)
data = pre.gender_update(data)
data = pre.degree_update(data)
data = pre.spec_update(data)
data = pre.collegeState_update(data)
data = pre.correct_values(data)

df_1 = data.iloc[:subset,:]
df_2 = data.iloc[subset:,:]

#tools.skew(df_2)
#tools.kurtosis(df_2)

y = df_2["Salary"]
x = df_2.drop(["Salary"], axis=1)

#tools.outliers(y)
'''
scaler1 = StandardScaler()
scaler2 = StandardScaler()
scaler1.fit(x)
y = y.array.reshape(-1,1)
scaler2.fit(y)
x = scaler1.transform(x)
y = scaler2.transform(y)
'''

t2.test_type2(x,y)

#x = x.drop(["GraduationYear","Specialization","extraversion","ComputerProgramming","CollegeCityTier", "Logical", "openess_to_experience"], axis=1)

salary_model = LinearRegression()
salary_model2 = LinearRegression()
'''
scaler1 = StandardScaler()
scaler2 = StandardScaler()
scaler1.fit(x)
y = y.array.reshape(-1,1)
scaler2.fit(y)
x = scaler1.transform(x)
y = scaler2.transform(y)
'''
salary_model.fit(x, y)
salary_r2 = salary_model.score(x, y)
print('R^2: {0}'.format(salary_r2))

'''
y_test = df_1["Salary"]
x_test = df_1.drop(["Salary"], axis=1)
print(x_test.head())
'''
x_test = x
y_test = y
name_save = 'type2'
tab=["GraduationYear","Specialization","extraversion","ComputerProgramming","CollegeCityTier", "Logical", "openess_to_experience"]
plt.figure(figsize=(20,10))
x=x_test.drop(tab, axis=1)
salary_model2.fit(x, y)

print(salary_model2.coef_)
'''
tools.linear_assumption(salary_model2, x, y_test)
tools.linear_assumption(salary_model, x_test, y_test)
plt.show()

plt.figure(figsize=(20,10))
x=x_test.drop(tab, axis=1)
tools.homoscedasticity_assumption(salary_model2, x, y_test)
tools.homoscedasticity_assumption(salary_model, x_test, y_test)
plt.show()


print("#############################################################################")
print("#############################################################################")
#print(tools.calculate_residuals(salary_model, x_test, y_test))
print("#############################################################################")
print("#############################################################################")
#tools.linear_assumption(salary_model, x_test, y_test)
print("#############################################################################")
print("#############################################################################")
#tools.normal_errors_assumption(salary_model, x_test, y_test)
print("#############################################################################")
print("#############################################################################")
tools.multicollinearity_assumption(salary_model, x_test, y_test)
print("#############################################################################")
print("#############################################################################")
tools.autocorrelation_assumption(salary_model, x_test, y_test)
print("#############################################################################")
print("#############################################################################")
#tools.homoscedasticity_assumption(salary_model, x_test, y_test)
print("#############################################################################")
print("#############################################################################")


'''