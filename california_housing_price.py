# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 15:53:32 2019

@author: Rao
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
housing = pd.read_csv(r"C:\Users\Rao\OneDrive - Oklahoma A and M System\handson-ml-master\handson-ml-master\datasets\housing\housing.csv")

import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


housing.info()

#let look at the only categorical variable "ocean_proximity"
housing["ocean_proximity"].value_counts()
housing.describe()
housing.isnull().sum()
housing["total_bedrooms"].describe()


housing.hist(bins=50, figsize=(20,15))

#looking at the hist there's an anamaly in:
#1. median income, house_median_age, median_house_value its capped.
#2. attributes are differently scaled, requires feature scaling.
#3. heavy tailed histograms

#time to split data in training and test sets
#Just randomly spliting the data may create some biases here we see 
#median income is an important atribute, having distrubution similar to the sample
# so first we will convert the continous median_income into categorical term

housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
  strat_train_set = housing.loc[train_index]
  strat_test_set = housing.loc[test_index]
  
for set in (strat_train_set, strat_test_set):
   set.drop(["income_cat"], axis=1, inplace=True)  
   
housing = strat_train_set.copy()

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

import seaborn as sns
sns.set(context="paper",font="monospace")
housing_corr_matrix = housing.corr()
#set the matplotlib figure
fig, axe = plt.subplots(figsize=(12,8))
#Generate color palettes 
cmap = sns.diverging_palette(220,10,center = "light", as_cmap=True)
#draw the heatmap
sns.heatmap(housing_corr_matrix,vmax=1,square =True, cmap=cmap,annot=True );


housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"] 

housing_corr_matrix = housing.corr()
housing_corr_matrix["median_house_value"].sort_values(ascending=False)
# few of the new has high correlation value like 

#data cleaning

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

#custom build automating the job of cleaning and tranformation
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer

from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
     def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
       self.add_bedrooms_per_room = add_bedrooms_per_room
     def fit(self, X, y=None):
         return self # nothing else to do
     def transform(self, X, y=None):
         rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
         population_per_household = X[:, population_ix] / X[:, household_ix]
         if self.add_bedrooms_per_room:
                 bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
                 return np.c_[X, rooms_per_household, population_per_household,
                              bedrooms_per_room]
         else:
                 return np.c_[X, rooms_per_household, population_per_household]
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values) 

housing_num = housing.drop("ocean_proximity", axis=1)
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
        
num_pipeline = Pipeline([
          
          ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
        
          ])      

from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelBinarizer

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('label_binarizer',CustomLabelBinarizer()),
    ])

from sklearn.pipeline import FeatureUnion
full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])
housing_prepared = full_pipeline.fit_transform(housing)

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

some_data = housing.iloc[:5]

some_labels = housing_labels.iloc[:5]

#some_data_prepared = full_pipeline.transform(some_data)
some_data_prepared = housing_prepared[:5]

some_data_prepared

print("Predictions:\t", lin_reg.predict(some_data_prepared))

print("Labels:\t\t", list(some_labels))

from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse
# RMSE value comes to be 68628.1981
# this value isn't great, the house median value ranges from 120K-265K, 
#this error is too high

# Let's try some more models

#decision tree

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse
#this value of RMSE is 0.0 which is a clear case of over fitting

from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
    
display_scores(tree_rmse_scores)
# after performing the cross validation we are getting a mean error of 71053.35

#peforming cross validation for linear regression
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
scoring="neg_mean_squared_error", cv=10)

lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)
# here it is 69052.46

#know lets try random froest regressor

from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
housing_predictions_forest = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse
# 22542.396440343684
from sklearn.model_selection import cross_val_score
scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-scores)

display_scores(forest_rmse_scores)
#RMSE value of 35542.4

#tweeking using gridsearchsv to get best set of parameter 
from sklearn.model_selection import GridSearchCV
param_grid = [
{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
{'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)
from sklearn.model_selection import GridSearchCV
param_grid = [
{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
{'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)

grid_search.best_params_
grid_search.best_estimator_

#thats how we got feature importance
#from sklearn.preprocessing import LabelEncoder
#encoder = LabelEncoder()
#housing_cat = housing["ocean_proximity"]
#housing_cat_encoded = encoder.fit_transform(housing_cat)
#housing_cat_encoded


#from sklearn.preprocessing import OneHotEncoder
#encoder = OneHotEncoder()
#housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))

#display these importance scores next to their corresponding attribute names:
#extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]

#cat_one_hot_attribs = list(encoder.classes_)
#attributes = num_attribs + extra_attribs + cat_one_hot_attribs
#sorted(zip(feature_importances, attributes), reverse=True)

final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
#Produced an RMSE of 37683.5



