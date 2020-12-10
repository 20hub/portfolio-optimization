import pandas as pd
from numpy import *
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt

# Ref: https://csdnnews.blog.csdn.net/article/details/107903204
# https://blog.csdn.net/weixin_46302487/article/details/105342812
# https://blog.csdn.net/qq_30868737/article/details/109164548
# https://blog.csdn.net/weixin_40159138/article/details/88920018

# Gain data and Use Date as index
GDP = pd.read_csv('data/GDP.csv')
GDP = GDP.set_index("DATE")
# GDP = mat(GDP.values)  # Turn to matrix #.values could be omitted

CBI = pd.read_csv('data/CBI.csv')
CBI = CBI.set_index("DATE")
# CBI = mat(CBI.values)

CPI = pd.read_csv('data/CPI.csv')
CPI = CPI.set_index("DATE")
# CPI = mat(CPI.values)

NETEXP = pd.read_csv('data/NETEXP.csv')
NETEXP = NETEXP.set_index("DATE")
# NETEXP = mat(NETEXP.values)

POP = pd.read_csv('data/POP.csv')
# POP = POP.set_index("DATE")

PPI = pd.read_csv('data/PPI.csv')
PPI = PPI.set_index("DATE")
# PPI = mat(PPI.values)

UNRATE = pd.read_csv('data/UNRATE.csv')
UNRATE = UNRATE.set_index("DATE")
# UNRATE = mat(UNRATE.values)

num = len(GDP)

# Concat all predictors into one matrix with a column full of 1
# M-1
preds = [CBI, CPI, NETEXP, PPI, UNRATE]
preds = pd.concat(preds, axis=1)
preds.columns = ['CBI', 'CPI', 'NETEXP', 'PPI', 'UNRATE']

varb = [CBI, CPI, NETEXP, PPI, UNRATE, GDP]
DATA = pd.concat(varb, axis=1)
DATA.columns = ['CBI', 'CPI', 'NETEXP', 'PPI', 'UNRATE', 'GDP']

# M-2
one = ones((num, 1))

X = hstack((one, CBI, CPI, NETEXP, PPI, UNRATE))  # The form of array
# print(pd.DataFrame(X).describe())  # Get the detailed info of these predictors
X = mat(X)
GDP = mat(GDP)  # change the form here

# Getting the coefficients
# M-1： Using Linear Algebra
beta = (X.T * X).I * X.T * GDP
print("The Coefficients are:")
print(beta)

# M-2： Using functions directly
formula = 'GDP ~ CBI + CPI + NETEXP + PPI + UNRATE'
mod_pre5 = ols(formula, data=DATA).fit()
print(mod_pre5.summary())
# reg = LinearRegression()  # from sklearn.linear_model import LinearRegression
# reg.fit(preds, GDP)
# print(reg.coef_)
# print(reg.intercept_)

# ANOVA
anova_results = anova_lm(mod_pre5)
print("ANOVA is:")
print(anova_results)


# VIF
# M-1: Define function where dataf is the whole dataset and col_o is column name
def vif(dataf, col_i):
    cols = list(dataf.columns)
    cols.remove(col_i)
    cols_ex_i = cols
    fmula = col_i + '~' + '+'.join(cols_ex_i)
    r2 = ols(fmula, dataf).fit().rsquared
    return 1. / (1. - r2)


test_data = DATA[['CBI', 'CPI', 'NETEXP', 'PPI', 'UNRATE']]  # or using preds directly
print("VIFs are:")
for i in test_data.columns:
    print(i, '\t', vif(dataf=test_data, col_i=i))

# M-2: Using function directly
vifs = pd.DataFrame()
X = pd.DataFrame(X)
vifs["features"] = X.columns
vifs["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vifs)

# Mallow's Cp
# (1) All predictors
SSE = sum(mod_pre5.resid ** 2)
sigma_M = SSE / (num - 5 - 1)
C_p = SSE / sigma_M - num + 2 * (5+1)
# print(C_p)


# (2) Using 4 predictors
def mallowcp(dataf, col_i, p):
    cols = list(dataf.columns)
    cols.remove(col_i)
    cols_ex_i = cols
    fmula = 'GDP' + '~' + '+'.join(cols_ex_i)
    md = ols(fmula, dataf).fit()
    sse = sum(md.resid ** 2)
    cp = sse / sigma_M - num + 2 * (p + 1)
    return [cp, col_i]

cp_4 = []
cp_3 = []
cp_2 = []
for i in test_data.columns:
    [cp4, col_i4] = mallowcp(dataf=test_data, col_i=i, p=4)
    cp_4.append(cp4)
    print("Excluding", i, ', the Cp is', cp4)
    for j in preds.drop(i, axis=1).columns:
        [cp3, col_i3] = mallowcp(dataf=preds.drop(i, axis=1), col_i=j, p=3)
        cp_3.append(cp3)
        # print("Excluding", i, "and", j, ', the Cp is', cp3)
        for k in preds.drop([i, j], axis=1).columns:
            [cp2, col_i2] = mallowcp(dataf=preds.drop([i, j], axis=1), col_i=k, p=2)
            cp_2.append(cp2)
            # print("Excluding", i, "and", j, "and", k, ', the Cp is', cp2)
cp_3 = list(set(cp_3))
cp_2 = list(set(cp_2))

for i in preds.columns:
    form = 'GDP ~' + i
    mod_pre1 = ols(form, data=DATA).fit()
    sse1 = sum(mod_pre1.resid ** 2)
    cp1 = sse1 / sigma_M - num + 2 * (1+1)
    print("With", i, ", the Cp is", cp1)

# Leverage Hii
H = mat(X) * (mat(X).T * mat(X)).I * mat(X).T  # Array
print(diag(H))
# print(sum(diag(H)))  # Verify the Leverage Property
plt.hist(diag(H), bins=20, edgecolor='b')
plt.axvline(2*6/44, color='r', linestyle='--')
plt.xlabel('Leverage Hii')
plt.ylabel('Counts')
plt.show()

# The raw residual, the studentized residual,  the externally studentized residual
raw_res = mod_pre5.resid  # Series
s = pd.DataFrame(raw_res).std()  # Series
s = list(s)[0]
stu_res = []
for i in range(44):
    stu_res.append(raw_res[i]/(s * sqrt(1 - diag(H)[i])))

remv_list = []
for i in range(44):
    orig = list(raw_res)
    orig.remove(orig[i])
    remv_list.append(orig)
s_ex_i = []
for i in range(44):
    s_withouti = list(pd.DataFrame(remv_list[i]).std())[0]
    s_ex_i.append(s_withouti)
ext_res = []
for i in range(44):
    ext_res.append(raw_res[i]/(s_ex_i[i] * sqrt(1 - diag(H)[i])))

plt.subplot(131)
plt.hist(raw_res, bins=20, edgecolor='b')
plt.xlabel('The Raw Residual')
plt.ylabel('Counts')

plt.subplot(132)
plt.hist(stu_res, bins=20, edgecolor='b')
plt.xlabel('The Studentized Residual')
plt.ylabel('Counts')

plt.subplot(133)
plt.hist(ext_res, bins=20, edgecolor='b')
plt.xlabel('The Externally Studentized Residual')
plt.ylabel('Counts')
plt.show()

# The Cook Distance
large_H = []
for i in range(44):
    if diag(H)[i] > 12/44:
        large_H.append(i)

date_exi = []
for i in range(len(large_H)):
    date_exi.append(DATA.index[i+1])
DATA_ex1 = DATA.drop(date_exi[0])
DATA_ex2 = DATA.drop(date_exi[1])
mod_ex_1 = ols(formula, data=DATA_ex1).fit()
mod_ex_2 = ols(formula, data=DATA_ex2).fit()
D1 = sum((mod_pre5.fittedvalues - mod_ex_1.fittedvalues) ** 2) / ((5+1) * (s ** 2))
D2 = sum((mod_pre5.fittedvalues - mod_ex_2.fittedvalues) ** 2) / ((5+1) * (s ** 2))
print("Cook's Distance without 2009-04-01 is:", D1)
print("Cook's Distance without 2009-07-01 is:", D2)
plt.plot(date_exi, [D1, D2], 'ro')
plt.xlabel("The Cook's Distance")
plt.show()

# Choose the model without PPI
formu4 = 'GDP ~ CBI + CPI + NETEXP + UNRATE'
mod_pre4 = ols(formu4, data=DATA).fit()

new_anova = anova_lm(mod_pre4)





