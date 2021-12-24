import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

import numpy as np
import statsmodels.api as sm

data = pd.read_csv("Engineering_graduate_salary.csv", sep=",")

length_data = len(data)
subset = round(length_data*.1)
print(subset)
print(length_data)
print(data.head())

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

df_1 = data.iloc[:subset,:]
df_2 = data.iloc[subset:,:]

###Hypothesis:
# The more people do skill tests, and the best their score on those tests and during their studies, the highest their salary.
# We will try to demonstrate (if their exist,) the correlation between school and ACMAT scores and salary.



column_names = []
for col in df_2.columns:
    column_names.append(col)

print(df_2.dtypes)
df_2['Gender'] = df_2['Gender'].astype('int64')
df_2['DOB'] = df_2['DOB'].astype('int64')
df_2['Degree'] = df_2['Degree'].astype('int64')
df_2['Specialization'] = df_2['Specialization'].astype('int64')
df_2['CollegeState'] = df_2['CollegeState'].astype('int64')

for i in range(len(df_2)):
    cols = ['Domain', 'ComputerProgramming', 'ElectronicsAndSemicon', 'ComputerScience', 'MechanicalEngg', 'ElectricalEngg', 'TelecomEngg', 'CivilEngg']
    for j in range(len(cols)):
        col = data.columns.get_loc(cols[j])
        content = df_2.iloc[i, col]
        if content < 0:
            df_2.iloc[i, col] = None

print(df_2.skew(axis = 0, skipna = True))
print(df_2.kurtosis(axis=None, skipna=None, level=None, numeric_only=None))


#x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
print(df_2.head())
df_2 = df_2.dropna(axis=0)
print("hello")
print(df_2.head())
y = df_2["Salary"]
#df_2.drop("Salary")
x = df_2
x, y = np.array(x), np.array(y)
x = sm.add_constant(x)
model = sm.OLS(y, x)
results = model.fit()
print(results.summary())
print('coefficient of determination:', results.rsquared)

print('adjusted coefficient of determination:', results.rsquared_adj)

print('regression coefficients:', results.params)


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

