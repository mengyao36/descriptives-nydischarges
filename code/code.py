### Data used: Hospital Inpatient Discharges (SPARCS De-Identified): 2016 ###
### Link: https://health.data.ny.gov/Health/Hospital-Inpatient-Discharges-SPARCS-De-Identified/gnzp-ekau ###


####################################################################
# Step 1 - import potentially needed packages #
from distutils.util import split_quoted
from itertools import groupby
import math
import statistics
from unittest import result

import numpy as np
import scipy as sc
from scipy import stats
from scipy.stats import lognorm
from scipy.stats import shapiro
from scipy.stats import chi2_contingency

import matplotlib.pyplot as plt

import pandas as pd
import pandas.plotting as plotting

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols

import seaborn as sns

import urllib
import urllib.request
import os

import bioinfokit
from bioinfokit.analys import stat

import tableone as t1
from tableone import TableOne

import researchpy as rp

import patsy

####################################################################
# Step 2 - load dataset 'sparcs' #
sparcs_original = pd.read_csv('data/sparcs_original.csv', low_memory=False)
# to fit the ram memory, select a random sample including 1,000 rows from the dataset #
sparcs = sparcs_original.sample(1000)
sparcs.to_csv('data/sparcs.csv')


####################################################################
# Step 3 - get some basic information of the dataset # 
# count total rows and columns in this dataset #
sparcs.shape # (row: 1,000, columns: 34)
# show all variables in this dataset #
sparcs.columns
# remove all white spaces/special characters in all variable names #
sparcs.columns = sparcs.columns.str.replace('[^A-Za-z0-9]+', '_')
# change all variable names into lower case #
sparcs.columns = sparcs.columns.str.lower()
# check editted variable names in a list #
for col_name in sparcs.columns: 
    print(col_name)
# check all variable types #
sparcs.dtypes
# check missing values in the dataset #
sparcs.isnull().sum()
# show the percentage of missing values for each variable #
sparcs.isnull().sum() *100 /len(sparcs) # payment_typology_2 (36%) & payment_typology_3 (76%) missing
# get a brief description of the dataset #
sparcs.describe()

# use 'tableone' to display basic information - group by race example (need to convert 'length_of_stay' to numerical first) # 
columns = ['age_group', 'gender', 'race', 'ethnicity', 'length_of_stay']
categorical = ['age_group', 'gender', 'race', 'ethnicity']
groupby = ['race']
mytable = TableOne(sparcs, columns=columns, categorical=categorical, groupby=groupby, pval=False)
print(mytable.tabulate(tablefmt = "fancy_grid"))
mytable.to_excel('data/tableone_example.xlsx')

# use 'researchpy' to display the codebook of dataset #
rp.codebook(sparcs)

####################################################################
# Step 4 - get descriptive data for some variables to generate a 'Table 1 - descriptive statistics' #
# var1: health_service_area #
#sparcs['health_service_area'].nunique()
print(sparcs['health_service_area'].unique())
print(type(sparcs['health_service_area'].unique()))
# count duplicates in var1 https://datatofish.com/count-duplicates-pandas/ #
health_service_area_cat = sparcs.pivot_table(columns=['health_service_area'], aggfunc='size')
print(health_service_area_cat)
# show the percentage of each category in the variable #
sparcs['health_service_area'].value_counts(normalize=True).sort_index() * 100

# var2: hospital_county #
hospital_county_cat = sparcs.pivot_table(columns=['hospital_county'], aggfunc='size')
print(hospital_county_cat)
# show the percentage of each category in the variable #
sparcs['hospital_county'].value_counts(normalize=True).sort_index() * 100

# var3: facility_name #
facility_name_cat = sparcs.pivot_table(columns=['facility_name'], aggfunc='size')
print(facility_name_cat)
# show the percentage of each category in the variable #
sparcs['facility_name'].value_counts(normalize=True).sort_index() * 100

# var4: age #
age_group_cat = sparcs.pivot_table(columns=['age_group'], aggfunc='size')
print(age_group_cat)
# show the percentage of each category in the variable #
sparcs['age_group'].value_counts(normalize=True).sort_index() * 100

# var5: gender #
gender_cat = sparcs.pivot_table(columns=['gender'], aggfunc='size')
print(gender_cat)
# show the percentage of each category in the variable #
sparcs['gender'].value_counts(normalize=True).sort_index() * 100

# var5: race #
race_cat = sparcs.pivot_table(columns=['race'], aggfunc='size')
print(race_cat)
# show the percentage of each category in the variable #
sparcs['race'].value_counts(normalize=True).sort_index() * 100

# var6: ethnicity #
ethnicity_cat = sparcs.pivot_table(columns=['ethnicity'], aggfunc='size')
print(ethnicity_cat)
# show the percentage of each category in the variable #
sparcs['ethnicity'].value_counts(normalize=True).sort_index() * 100

# var7: length_of_stay - continous variable #
# remove all special characters and whitespace from this var # 
sparcs['length_of_stay'] = sparcs['length_of_stay'].str.replace('[^A-Za-z0-9]+', '')
# convert this var from str to float #
sparcs['length_of_stay'] = sparcs['length_of_stay'].astype(float)
print(sparcs['length_of_stay'].dtypes)
sparcs['total_charges'].head(5)
# get mean, sd, min, max of this var #
sparcs['length_of_stay'].mean()
sparcs['length_of_stay'].std()
sparcs['length_of_stay'].min() #  min = 1.0
sparcs['length_of_stay'].max() # max = 120.0
# use 'researchpy' to fastern the steps above #
rp.summary_cont(sparcs['length_of_stay']) # mean = 5.40, sd = 8.14
# check if this var is normally distributed # 
# check skewness #
sparcs['length_of_stay'].skew() # skew = 6.79 > 0, left tail/right skewed 
# check q-q plot #
np.random.seed(1)
fig = sm.qqplot(sparcs['length_of_stay'], line='45')
plt.show()
# perform shapiro-wilk test - p > 0.05 means normally distributed #
np.random.seed(1)
shapiro(sparcs['length_of_stay']) # p < 0.05, not normally distributed 

# var8: type_of_admission #
type_of_admission_cat = sparcs.pivot_table(columns=['type_of_admission'], aggfunc='size')
print(type_of_admission_cat)
# show the percentage of each category in the variable #
sparcs['type_of_admission'].value_counts(normalize=True).sort_index() * 100

# var9: patient_disposition #
patient_disposition_cat = sparcs.pivot_table(columns=['patient_disposition'], aggfunc='size')
print(patient_disposition_cat)
# show the percentage of each category in the variable #
sparcs['patient_disposition'].value_counts(normalize=True).sort_index() * 100

# var10: ccs_diagnosis_description #
ccs_diagnosis_description_cat = sparcs.pivot_table(columns=['ccs_diagnosis_description'], aggfunc='size')
print(ccs_diagnosis_description_cat)
# show the percentage of each category in the variable #
sparcs['ccs_diagnosis_description'].value_counts(normalize=True).sort_index() * 100

# var11: apr_severity_of_illness_code #
apr_severity_of_illness_code_cat = sparcs.pivot_table(columns=['apr_severity_of_illness_code'], aggfunc='size')
print(apr_severity_of_illness_code_cat)
# show the percentage of each category in the variable #

# var12: apr_severity_of_illness_description #
apr_severity_of_illness_description_cat = sparcs.pivot_table(columns=['apr_severity_of_illness_description'], aggfunc='size')
print(apr_severity_of_illness_description_cat)
# show the percentage of each category in the variable #
sparcs['apr_severity_of_illness_description'].value_counts(normalize=True).sort_index() * 100

# var13: apr_risk_of_mortality #
apr_risk_of_mortality_cat = sparcs.pivot_table(columns=['apr_risk_of_mortality'], aggfunc='size')
print(apr_risk_of_mortality_cat)
# show the percentage of each category in the variable #
sparcs['apr_risk_of_mortality'].value_counts(normalize=True).sort_index() * 100

# var14: payment_typology_1 #
payment_typology_1_cat = sparcs.pivot_table(columns=['payment_typology_1'], aggfunc='size')
print(payment_typology_1_cat)
# show the percentage of each category in the variable #
sparcs['payment_typology_1'].value_counts(normalize=True).sort_index() * 100

# var15: payment_typology_2 #
payment_typology_2_cat = sparcs.pivot_table(columns=['payment_typology_2'], aggfunc='size')
print(payment_typology_2_cat)
# show the percentage of each category in the variable #
sparcs['payment_typology_2'].value_counts(normalize=True).sort_index() * 100

# var16: payment_typology_3 #
payment_typology_3_cat = sparcs.pivot_table(columns=['payment_typology_3'], aggfunc='size')
print(payment_typology_3_cat)
# show the percentage of each category in the variable #
sparcs['payment_typology_3'].value_counts(normalize=True).sort_index() * 100

# var17: total_charges - continous variable #
# remove all special characters and whitespace from this var # 
sparcs['total_charges'] = sparcs['total_charges'].str.replace('[^A-Za-z0-9]+', '_')
# convert this var from str to float #
sparcs['total_charges'] = sparcs['total_charges'].astype(float)
print(sparcs['total_charges'].dtypes)
sparcs['total_charges'].head(5)
# get mean, sd, min, max of this var #
sparcs['total_charges'].mean()
sparcs['total_charges'].std()
sparcs['total_charges'].min() # min = 2039.0
sparcs['total_charges'].max() # max = 192194371.0
# use 'researchpy' to fastern the steps above #
rp.summary_cont(sparcs['total_charges']) # mean = 3768452.22, sd = 9.05 
# check if this var is normally distributed # 
# check skewness #
sparcs['total_charges'].skew() # skew = 12.31 > 0, left tail/right skewed 
# check q-q plot #
np.random.seed(1)
fig = sm.qqplot(sparcs['total_charges'], line='45')
plt.show()
# perform shapiro-wilk test - p > 0.05 means normally distributed #
np.random.seed(1)
shapiro(sparcs['total_charges']) # p < 0.05, not normally distribuited 

# var18: total_costs - continous variable#
# remove all special characters and whitespace from this var # 
sparcs['total_costs'] = sparcs['total_costs'].str.replace('[^A-Za-z0-9]+', '_')
# convert this var from str to float #
sparcs['total_costs'] = sparcs['total_costs'].astype(float)
print(sparcs['total_costs'].dtypes)
sparcs['total_costs'].head(5)
# get mean, sd, min, max of this var #
sparcs['total_costs'].mean()
sparcs['total_costs'].std()
sparcs['total_costs'].min() # min = 2547.0
sparcs['total_costs'].max() # max = 86043492.0 
# use 'researchpy' to fastern the steps above #
rp.summary_cont(sparcs['total_costs']) # mean = 1449261.16, sd = 3.76
# check if this var is normally distributed # 
# check skewness #
sparcs['total_costs'].skew() # skew = 15.15 > 0, left tail/right skewed 
# check q-q plot #
np.random.seed(1)
fig = sm.qqplot(sparcs['total_costs'], line='45')
plt.show()
# perform shapiro-wilk test - p > 0.05 means normally distributed #
np.random.seed(1)
shapiro(sparcs['total_costs']) # p < 0.05, not normally distribuited


####################################################################
### Step 5 - get correlation between some variables ###
# Pearson's correlation efficient test - example #
# assess the relationship between the length of stay in the hospital and patient's total charges #
sparcs['length_of_stay'].corr(sparcs['total_charges']) # corr = 0.69
# assess the relationship between the length of stay in the hospital and patient's total costs #
sparcs['length_of_stay'].corr(sparcs['total_costs']) # corr = 0.71
# assess the relationship between patient's total charges and patient's total costs #
sparcs['total_charges'].corr(sparcs['total_costs']) # corr = 0.92

# 2-sample t-test - example #
# assess the mean difference of length_of_stay across gender groups #
female_stay = sparcs[sparcs['gender'] == 'F']['length_of_stay']
male_stay = sparcs[sparcs['gender'] == 'M']['length_of_stay']
other_stay = sparcs[sparcs['gender'] == 'U']['length_of_stay']
stats.ttest_ind(female_stay, male_stay) # p = 0.16 , not sig
stats.ttest_ind(female_stay, other_stay) # p = nan
stats.ttest_ind(male_stay, other_stay) # p = nan 
# assess the mean difference of length_of_stay across race groups #
black_stay = sparcs[sparcs['race'] == 'Black/African American']['length_of_stay']
multi_stay = sparcs[sparcs['race'] == 'Multi-racial']['length_of_stay']
other_stay = sparcs[sparcs['race'] == 'Other Race']['length_of_stay']
white_stay = sparcs[sparcs['race'] == 'White']['length_of_stay']
stats.ttest_ind(black_stay, multi_stay) # p = 1.16, not sig
stats.ttest_ind(black_stay, other_stay) # p = 0.57, not sig 
stats.ttest_ind(black_stay, white_stay) # p = 0.07, not sig 
stats.ttest_ind(multi_stay, other_stay) # p = 0.41, not sig 
stats.ttest_ind(multi_stay, white_stay) # p = 0.20, not sig
stats.ttest_ind(other_stay, white_stay) # p = 0.47, not sig 

# Liner Regression Model #
# Assess the relationship between patient's age and length of stay #
# convert 'age_group' to dummy variable https://www.statology.org/pandas-data-cast-to-numpy-dtype-of-object-check-input-data-with-np-asarraydata/ #
sparcs = pd.get_dummies(sparcs, columns=['age_group'], drop_first=True)
sparcs.columns
x = sparcs['age_group_18 to 29']
y = sparcs['length_of_stay']
model = sm.OLS(y, x).fit()
predictions = model.predict(x)
model.summary() # coef = 3.86, se = 0.96, p < 0.01, sig 


####################################################################
### Step 6 - visulization ###
# simple plotting: show the relationship between patient's length of stay and total charges  plot 2 linear fits for male and female #
sns.lmplot(y='total_charges', x='length_of_stay', hue='gender', data=sparcs)
plt.show()