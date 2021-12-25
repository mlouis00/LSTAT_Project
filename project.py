import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

data = pd.read_csv("Engineering_graduate_salary.csv", sep=",")

length_data = len(data)
subset = round(length_data*.1)
#print(subset)
#print(length_data)
#print(data.head())

data = data.drop(["ID", "10board", "12board"], axis=1) # College ID useless?

def date_update():
    col = data["DOB"]
    i = 0
    end = len(data)
    DOB = data.columns.get_loc("DOB")
    while i < end:
        year = col[i][:4]
        data.iloc[i, DOB] = year
        i += 1

def gender_update():
    col = data["Gender"]
    i = 0
    end = len(data)
    gender_col = data.columns.get_loc("Gender")
    while i < end:
        gender = 0
        if col[i] == 'f':
            gender = 1
        data.iloc[i, gender_col] = gender
        i += 1

def degree_update():
    col = data["Degree"]
    i = 0
    end = len(data)
    degree_col = data.columns.get_loc("Degree")
    while i < end:
        degree = 0
        cell_info = col[i]
        if cell_info == 'B.Tech/B.E.':
            degree = 0
        elif cell_info == 'M.Tech./M.E.':
            degree = 1
        elif cell_info == 'MCA':
            degree = 2
        elif cell_info == 'M.Sc. (Tech.)':
            degree = 3
        data.iloc[i, degree_col] = degree
        i += 1

def spec_update():
    col = data["Specialization"]
    spec_col = data.columns.get_loc("Specialization")
    spec_names = []
    for i in range(len(col)):
        el = data.iloc[i, spec_col]
        if el in spec_names:
            idx = spec_names.index(el)
            data.iloc[i, spec_col] = idx
        else:
            idx = len(spec_names)
            spec_names.append(el)
            data.iloc[i, spec_col] = idx

def collegeState_update():
    col = data["CollegeState"]
    collegeState_col = data.columns.get_loc("CollegeState")
    collegeState_names = []
    for i in range(len(col)):
        el = data.iloc[i, collegeState_col]
        if el in collegeState_names:
            idx = collegeState_names.index(el)
            data.iloc[i, collegeState_col] = idx
        else:
            idx = len(collegeState_names)
            collegeState_names.append(el)
            data.iloc[i, collegeState_col] = idx


date_update()
gender_update()
degree_update()
spec_update()
collegeState_update()

column_names = []
for col in data.columns:
    column_names.append(col)

print(data.dtypes)
data['Gender'] = data['Gender'].astype('int64')
data['DOB'] = data['DOB'].astype('int64')
data['Degree'] = data['Degree'].astype('int64')
data['Specialization'] = data['Specialization'].astype('int64')
data['CollegeState'] = data['CollegeState'].astype('int64')

for i in range(len(data)):
    cols = ['GraduationYear', 'Domain', 'ComputerProgramming', 'ElectronicsAndSemicon', 'ComputerScience', 'MechanicalEngg', 'ElectricalEngg', 'TelecomEngg', 'CivilEngg']
    for j in range(len(cols)):
        col = data.columns.get_loc(cols[j])
        content = data.iloc[i, col]
        if j == 0:
          if content <= 0:
            data.iloc[i, col] = None  
        else:
            if content < 0:
                data.iloc[i, col] = None

data = data.fillna(-9999)
df_1 = data.iloc[:subset,:]
df_2 = data.iloc[subset:,:]

###Hypothesis:
# The more people do skill tests, and the best their score on those tests and during their studies, the highest their salary.
# We will try to demonstrate (if their exist,) the correlation between school and ACMAT scores and salary.


print(df_2.skew(axis = 0, skipna = True))
print(df_2.kurtosis(axis=None, skipna=None, level=None, numeric_only=None))

y = df_2["Salary"]
print(y)
x = df_2.drop(["Salary"], axis=1)
print(x.head())
'''
x, y = np.array(x), np.array(y)
x = sm.add_constant(x)
model = sm.OLS(y, x)
results = model.fit()
print(results.summary())
print('coefficient of determination:', results.rsquared)

print('adjusted coefficient of determination:', results.rsquared_adj)

print('regression coefficients:', results.params)
'''
############### Sklearn

salary_model = LinearRegression()
salary_model.fit(x, y)
salary_r2 = salary_model.score(x, y)
print('R^2: {0}'.format(salary_r2))

def calculate_residuals(model, features, label):
    """
    Creates predictions on the features with the model and calculates residuals
    """
    predictions = model.predict(features)
    df_results = pd.DataFrame({'Actual': label, 'Predicted': predictions})
    df_results['Residuals'] = abs(df_results['Actual']) - abs(df_results['Predicted'])
    
    return df_results

def linear_assumption(model, features, label):
    """
    Linearity: Assumes that there is a linear relationship between the predictors and
               the response variable. If not, either a quadratic term or another
               algorithm should be used.
    """
    print('Assumption 1: Linear Relationship between the Target and the Feature', '\n')
        
    print('Checking with a scatter plot of actual vs. predicted.',
           'Predictions should follow the diagonal line.')
    
    # Calculating residuals for the plot
    df_results = calculate_residuals(model, features, label)
    
    # Plotting the actual vs predicted values
    sns.lmplot(x='Actual', y='Predicted', data=df_results, fit_reg=False, size=7)
        
    # Plotting the diagonal line
    line_coords = np.arange(df_results.min().min(), df_results.max().max())
    plt.plot(line_coords, line_coords,  # X and y points
             color='darkorange', linestyle='--')
    plt.title('Actual vs. Predicted')
    plt.show()

def normal_errors_assumption(model, features, label, p_value_thresh=0.05):
    """
    Normality: Assumes that the error terms are normally distributed. If they are not,
    nonlinear transformations of variables may solve this.
               
    This assumption being violated primarily causes issues with the confidence intervals
    """
    from statsmodels.stats.diagnostic import normal_ad
    print('Assumption 2: The error terms are normally distributed', '\n')
    
    # Calculating residuals for the Anderson-Darling test
    df_results = calculate_residuals(model, features, label)
    
    print('Using the Anderson-Darling test for normal distribution')

    # Performing the test on the residuals
    p_value = normal_ad(df_results['Residuals'])[1]
    print('p-value from the test - below 0.05 generally means non-normal:', p_value)
    
    # Reporting the normality of the residuals
    if p_value < p_value_thresh:
        print('Residuals are not normally distributed')
    else:
        print('Residuals are normally distributed')
    
    # Plotting the residuals distribution
    plt.subplots(figsize=(12, 6))
    plt.title('Distribution of Residuals')
    sns.distplot(df_results['Residuals'])
    plt.show()
    
    print()
    if p_value > p_value_thresh:
        print('Assumption satisfied')
    else:
        print('Assumption not satisfied')
        print()
        print('Confidence intervals will likely be affected')
        print('Try performing nonlinear transformations on variables')

def multicollinearity_assumption(model, features, label, feature_names=None):
    """
    Multicollinearity: Assumes that predictors are not correlated with each other. If there is
                       correlation among the predictors, then either remove prepdictors with high
                       Variance Inflation Factor (VIF) values or perform dimensionality reduction
                           
                       This assumption being violated causes issues with interpretability of the 
                       coefficients and the standard errors of the coefficients.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    print('Assumption 3: Little to no multicollinearity among predictors')
        
    # Plotting the heatmap
    plt.figure(figsize = (10,8))
    sns.heatmap(pd.DataFrame(features, columns=feature_names).corr(), annot=True)
    plt.title('Correlation of Variables')
    plt.show()
        
    print('Variance Inflation Factors (VIF)')
    print('> 10: An indication that multicollinearity may be present')
    print('> 100: Certain multicollinearity among the variables')
    print('-------------------------------------')
       
    # Gathering the VIF for each variable
    VIF = [variance_inflation_factor(features, i) for i in range(features.shape[1])]
    #for idx, vif in enumerate(VIF):
        #print('{0}: {1}'.format(feature_names[idx], vif))
        
    # Gathering and printing total cases of possible or definite multicollinearity
    possible_multicollinearity = sum([1 for vif in VIF if vif > 10])
    definite_multicollinearity = sum([1 for vif in VIF if vif > 100])
    print()
    print('{0} cases of possible multicollinearity'.format(possible_multicollinearity))
    print('{0} cases of definite multicollinearity'.format(definite_multicollinearity))
    print()

    if definite_multicollinearity == 0:
        if possible_multicollinearity == 0:
            print('Assumption satisfied')
        else:
            print('Assumption possibly satisfied')
            print()
            print('Coefficient interpretability may be problematic')
            print('Consider removing variables with a high Variance Inflation Factor (VIF)')

    else:
        print('Assumption not satisfied')
        print()
        print('Coefficient interpretability will be problematic')
        print('Consider removing variables with a high Variance Inflation Factor (VIF)')

def autocorrelation_assumption(model, features, label):
    """
    Autocorrelation: Assumes that there is no autocorrelation in the residuals. If there is
                     autocorrelation, then there is a pattern that is not explained due to
                     the current value being dependent on the previous value.
                     This may be resolved by adding a lag variable of either the dependent
                     variable or some of the predictors.
    """
    from statsmodels.stats.stattools import durbin_watson
    print('Assumption 4: No Autocorrelation', '\n')
    
    # Calculating residuals for the Durbin Watson-tests
    df_results = calculate_residuals(model, features, label)

    print('\nPerforming Durbin-Watson Test')
    print('Values of 1.5 < d < 2.5 generally show that there is no autocorrelation in the data')
    print('0 to 2< is positive autocorrelation')
    print('>2 to 4 is negative autocorrelation')
    print('-------------------------------------')
    durbinWatson = durbin_watson(df_results['Residuals'])
    print('Durbin-Watson:', durbinWatson)
    if durbinWatson < 1.5:
        print('Signs of positive autocorrelation', '\n')
        print('Assumption not satisfied')
    elif durbinWatson > 2.5:
        print('Signs of negative autocorrelation', '\n')
        print('Assumption not satisfied')
    else:
        print('Little to no autocorrelation', '\n')
        print('Assumption satisfied')
        
def homoscedasticity_assumption(model, features, label):
    """
    Homoscedasticity: Assumes that the errors exhibit constant variance
    """
    print('Assumption 5: Homoscedasticity of Error Terms', '\n')
    
    print('Residuals should have relative constant variance')
        
    # Calculating residuals for the plot
    df_results = calculate_residuals(model, features, label)

    # Plotting the residuals
    plt.subplots(figsize=(12, 6))
    ax = plt.subplot(111)  # To remove spines
    plt.scatter(x=df_results.index, y=df_results.Residuals, alpha=0.5)
    plt.plot(np.repeat(0, df_results.index.max()), color='darkorange', linestyle='--')
    ax.spines['right'].set_visible(False)  # Removing the right spine
    ax.spines['top'].set_visible(False)  # Removing the top spine
    plt.title('Residuals')
    plt.show()  

y_test = df_1["Salary"]
x_test = df_1.drop(["Salary"], axis=1)
print(x_test.head())
#calculate_residuals(salary_model, x_test, y_test)
print("#############################################################################")
print("#############################################################################")
print("#############################################################################")
print("#############################################################################")
#linear_assumption(salary_model, x_test, y_test)
print("#############################################################################")
print("#############################################################################")
print("#############################################################################")
print("#############################################################################")
#normal_errors_assumption(salary_model, x_test, y_test)
print("#############################################################################")
print("#############################################################################")
print("#############################################################################")
print("#############################################################################")
multicollinearity_assumption(salary_model, x_test, y_test)
print("#############################################################################")
print("#############################################################################")
print("#############################################################################")
print("#############################################################################")
autocorrelation_assumption(salary_model, x_test, y_test)
print("#############################################################################")
print("#############################################################################")
print("#############################################################################")
print("#############################################################################")
homoscedasticity_assumption(salary_model, x_test, y_test)
print("#############################################################################")
print("#############################################################################")
print("#############################################################################")
print("#############################################################################")

"""
print(df_2.describe())
description = df_2.describe()
for i in description:
    print(i)
    for j in range(len(description[i])):
        if j == 1 or j == 2:
            print(round(description[i][j], 2))



df_2.describe().to_csv('description_dataset.csv', index=True)
print(df_2.head())

corrMatrix = df_2.corr()
corrMatrix = corrMatrix.sort_values(by=['Salary'], ascending=False)
corrMatrix_bis = corrMatrix.copy()
print(corrMatrix)
sn.heatmap(corrMatrix, annot=True)
plt.show()
cols = len(corrMatrix)
rows = len(corrMatrix)
col = 0
row = 0
while col < cols:
    while row < rows:
        el = corrMatrix[column_names[col]][column_names[row]]
        res = -1
        if el < 0:
            res = -0.2
        elif el < 0.2 and el >= 0:
            res = 0
        elif el < 0.4 and el >= 0.2:
            res = 0.2
        elif el < 0.6 and el >= 0.4:
            res = 0.4
        elif el < 0.8 and el >= 0.6:
            res = 0.6
        elif el < 1 and el >= 0.8:
            res = 0.8
        elif el == 1:
            res = 1
        #print(el, "###", res)
        corrMatrix_bis[column_names[col]][column_names[row]] = res
        row += 1
    row = 0
    col += 1


corrMatrix.to_csv('data_corr.csv', index=True)
corrMatrix_bis.to_csv('data_corr_bis.csv', index=True)

sn.heatmap(corrMatrix_bis, annot=True)
plt.show()

print(df_2.columns)
column_names = []
for col in df_2.columns:
    #column_names.append(col)
    boxplot = df_2.boxplot(column=col)
    plt.show() 


"""

