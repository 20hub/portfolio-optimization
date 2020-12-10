import pandas as pd
from numpy import *
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


# Gain data and Use Date as index
GDP = pd.read_csv('data/GDP.csv')
GDP = GDP.set_index("DATE")
CBI = pd.read_csv('data/CBI.csv')
CBI = CBI.set_index("DATE")
CPI = pd.read_csv('data/CPI.csv')
CPI = CPI.set_index("DATE")
NETEXP = pd.read_csv('data/NETEXP.csv')
NETEXP = NETEXP.set_index("DATE")
PPI = pd.read_csv('data/PPI.csv')
PPI = PPI.set_index("DATE")
UNRATE = pd.read_csv('data/UNRATE.csv')
UNRATE = UNRATE.set_index("DATE")

GDPpre = pd.read_csv('data/GDPpre.csv')
GDPpre = GDPpre.set_index("DATE")
CBIpre = pd.read_csv('data/CBIpre.csv')
CBIpre = CBIpre.set_index("DATE")
CPIpre = pd.read_csv('data/CPIpre.csv')
CPIpre = CPIpre.set_index("DATE")
NETEXPpre = pd.read_csv('data/NETEXPpre.csv')
NETEXPpre = NETEXPpre.set_index("DATE")
UNRATEpre = pd.read_csv('data/UNRATEpre.csv')
UNRATEpre = UNRATEpre.set_index("DATE")

varb = [CBI, CPI, NETEXP, PPI, UNRATE, GDP]
DATA = pd.concat(varb, axis=1)
DATA.columns = ['CBI', 'CPI', 'NETEXP', 'PPI', 'UNRATE', 'GDP']

newpreds = [CBIpre, CPIpre, NETEXPpre, UNRATEpre]
newpreds = pd.concat(newpreds, axis=1)
newpreds.columns = ['CBI', 'CPI', 'NETEXP', 'UNRATE']

newvarb = [CBI, CPI, NETEXP, UNRATE, GDP]
newDATA = pd.concat(newvarb, axis=1)
newDATA.columns = ['CBI', 'CPI', 'NETEXP', 'UNRATE', 'GDP']

# Choose the model without PPI
formu4 = 'GDP ~ CBI + CPI + NETEXP + UNRATE'
mod_pre4 = ols(formu4, data=DATA).fit()
print(mod_pre4.summary())

new_anova = anova_lm(mod_pre4)
print("New ANOVA is:")
print(new_anova)

new1 = list(newpreds.iloc[0])
new1.insert(0, 1)
GDP1 = mat(mod_pre4.params) * mat(new1).T
new1.remove(1)
new1.append(GDP1.tolist()[0][0])

new1 = pd.DataFrame(new1, index=newDATA.columns)
new1 = pd.DataFrame(new1.values.T, columns=new1.index, index=new1.columns)

d_ex = newDATA.drop(newDATA.index[0])
d1 = d_ex.append(new1)

formu4_2 = 'GDP ~ CBI + CPI + NETEXP + UNRATE'
mod_pre4_2 = ols(formu4_2, data=DATA).fit()

new2 = list(newpreds.iloc[1])
new2.insert(0, 1)
GDP2 = mat(mod_pre4_2.params) * mat(new2).T
new2.remove(1)
new2.append(GDP2.tolist()[0][0])

new2 = pd.DataFrame(new2, index=newDATA.columns)
new2 = pd.DataFrame(new2.values.T, columns=new2.index, index=new2.columns)

d_ex_2 = d1.drop(d1.index[0])
d2 = d_ex_2.append(new2)

formu4_3 = 'GDP ~ CBI + CPI + NETEXP + UNRATE'
mod_pre4_3 = ols(formu4_3, data=DATA).fit()

new3 = list(newpreds.iloc[2])
new3.insert(0, 1)
GDP3 = mat(mod_pre4_3.params) * mat(new3).T
new3.remove(1)
new3.append(GDP3.tolist()[0][0])

new3 = pd.DataFrame(new3, index=newDATA.columns)
new3 = pd.DataFrame(new3.values.T, columns=new3.index, index=new3.columns)

d_ex_3 = d2.drop(d2.index[0])
d3 = d_ex_3.append(new3)
print(d3)
