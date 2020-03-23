#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 20:46:24 2019

@author: cursedomonstro
"""


import os
import tarfile
from six.moves import urllib
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from pandas.tools.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection  import cross_val_score, GridSearchCV, RandomizedSearchCV

class CustomBinarizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None,**fit_params):
        return self
    def transform(self, X):
        return LabelBinarizer().fit(X).transform(X)

    
class DataFrameSelector(BaseEstimator,TransformerMixin):
  def __init__(self, attribute_names):
    self.attribute_names = attribute_names
  def fit(self,X,y=None)  :
    return self
  def transform(self,X):
    return X[self.attribute_names].values

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedroom_per_room=True):
        self.add_bedroom_per_room = add_bedroom_per_room

    
    def fit(self, X, y=None):
        return self #return identique object
    
    def transform(self, X, y=None):
        #bedroom_per_room = np.asarray(housing['total_bedrooms']/housing['total_rooms'])
        room_per_house = np.asarray(housing['total_rooms']/housing['households'])
        population_per_house = np.asarray(housing['population']/housing['households'])
        return np.c_[X, room_per_house, population_per_house]


       

DOWNLOAD_ROOT="https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH="datasets/housing"
HOUSING_URL=DOWNLOAD_ROOT+HOUSING_PATH+"/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL,housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path=os.path.join(housing_path,"housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz=tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    

import pandas as pd
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path=os.path.join(housing_path,"housing.csv")
    return pd.read_csv(csv_path)

import matplotlib.pyplot as plt
################## Debut de l'ecriture des fonctions ########################
fetch_housing_data() # telecharger les donnees les stocker
housing=load_housing_data()


import numpy as np

def split_train_test(data,test_ratio):
    shuffled_indices=np.random.permutation(len(data))
    test_indices=shuffled_indices[:int(test_ratio*len(data))]
    train_indices=shuffled_indices[int(test_ratio*len(data)):]
    return data.iloc[train_indices],data.iloc[test_indices]

# la donnee concernant le revenu moyen etant important on procede
# a une stratification
housing['income_cat'] = np.ceil(housing['median_income']/1.5)
housing['income_cat'].where(housing['income_cat']<5, 5, inplace=True)

#### plot the curve
#(housing['income_cat'].value_counts().sort_values(ascending=True)/housing.shape[0]).plot(kind='bar')

##### spliting the dataset into test and train dataset
#train_set,test_set=train_test_split(housing,test_size=0.2, random_state=42)


#### using a stratified splitting
"""split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_index = housing.loc[train_index]
    strat_test_index = housing.loc[test_index]"""
    
    
######## visualisation des donnees geographiques
"""housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)"""

#### visualisation plus complete des donnees
"""housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing['population']/100, label="population",
             c="median_house_value", cmap=plt.get_cmap("jet"),
             colorbar=True)"""

######" recherche de correlation dans entre la variable cible et les autres
"""corr_matrice = housing.corr()
print(corr_matrice['median_house_value'].sort_values(ascending=False))"""

# la valeur la plus liée a la variable a predire est le revenu moyen avec un coef
# de pearson de 0.68

######### regarder les correlations entre les differentes variables
"""attributes = ["median_house_value", "median_income",
              "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))"""

##### experimenter les combinaisons de variable
"""housing['bedroom_per_room'] = housing['total_bedrooms']/housing['total_rooms']
housing['room_per_house'] = housing['total_rooms']/housing['households']
housing['population_per_house'] = housing['population']/housing['households']

corr_matrice = housing.corr()
print(corr_matrice['median_house_value'].sort_values(ascending=False))"""
# less bedrooms per rooms for most expensive houses perhaps less people


#### Nettoyer les donnees

# Regarder les columnes avec les donnees manquantes
#print(housing.isnull().mean())
### only total_bedrooms have missing values 
## I can replace it by the mean, the median or 0
## Let's try with median
"""housing_copy = housing.copy()
tot_bed = housing['total_bedrooms'].median()
housing_copy['total_bedrooms'].fillna(tot_bed, inplace=True)
housing_copy = housing_copy.drop('median_house_value', axis=1)"""
#### delete the targeted column
"""housing_3 = housing.dropna(subset=['total_bedrooms'])"""

##### Methode plus evoluee avec sklearn
"""median_imputer = SimpleImputer(strategy="median")
housing_num = housing.select_dtypes(include=['number'])
median_imputer.fit(housing_num)
housing_med = median_imputer.transform(housing_num)
housing_tr = pd.DataFrame(housing_med, columns = housing_num.columns)"""

## Gestion des differents cas de OneHotEncoder 
"""housing_cat = housing.select_dtypes(exclude=['number'])
proximity_encoder = LabelBinarizer() #remplacer par label au besoin
X = proximity_encoder.fit_transform(housing_cat)
X = pd.DataFrame(X)"""


####### test pour le pipeline
"""columnes = ['room_per_house', 'population_per_house', 'bedroom_per_room']
total_columnes = list(housing.columns)+columnes
one_combiner = CombinedAttributesAdder(True)
v = one_combiner.fit_transform(housing)
new_dataframe = pd.DataFrame(v, columns= total_columnes)
new_dataframe = new_dataframe.drop('ocean_proximity', axis=1)

for col in new_dataframe.columns:
    new_dataframe[col] = new_dataframe[col].astype(float)
"""





##### using of linear regression (methode des moindres carrees)
"""model = LinearRegression()
model.fit(housing_final, housing_label)
some_data = housing_final[:5]
pred = model.predict(some_data)
print(np.sqrt(mean_squared_error(pred, housing_label[:5])))"""

#####" use of a decision tree regressor

"""model = DecisionTreeRegressor()
scores = cross_val_score(model, housing_final, housing_label, scoring="neg_mean_squared_error", cv=10)
print(np.sqrt(-scores))"""

###random forest regressor


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_index = housing.loc[train_index]
    strat_test_index = housing.loc[test_index]

strat_train_index = strat_train_index.drop('median_income', axis=1)
strat_train_label = strat_train_index['median_house_value']
strat_train_index = strat_train_index.drop('median_house_value', axis=1)

strat_test_index = strat_test_index.drop('median_income', axis=1)
strat_test_label = strat_test_index['median_house_value']
strat_test_index = strat_test_index.drop('median_house_value', axis=1)

num_attributs = list(strat_train_index.select_dtypes(include=['number']).columns)
cat_attributs = list(strat_train_index.select_dtypes(exclude=['number']).columns)


num_pipeline = Pipeline([('num_selector', DataFrameSelector(num_attributs)),
                         ('imputer', SimpleImputer(strategy='median')),
                         ('std_scaler', StandardScaler())
                        ])

cat_pipeline =  Pipeline([('cat_selector', DataFrameSelector(cat_attributs)),
                          ('label_binarizer', CustomBinarizer())
                          ])
final_pipeline = FeatureUnion(transformer_list=[('num_pipe', num_pipeline),
                              ('cat_pipe', cat_pipeline)
                              ])
    
housing_final = final_pipeline.fit_transform(strat_train_index)    

model = RandomForestRegressor()
param_grid = [{'n_estimators': [3,10,30], 'max_features':[2,4,6,8]},
              {'bootstrap':[False], 'n_estimators':[3,10],
                'max_features':[2,3,4]}
            ]
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(housing_final, strat_train_label)

final_model = grid_search.best_estimator_

test = final_pipeline.fit_transform(strat_test_index)
final_predictions = final_model.predict(test)
final_rmse = np.sqrt(mean_squared_error(strat_test_label, final_predictions))
print(final_rmse)