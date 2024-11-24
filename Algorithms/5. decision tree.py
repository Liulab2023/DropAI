#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


data = pd.read_csv(r'DATApep-3.csv',parse_dates=[3])
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
X_train = train_set.drop(columns=['Y'], axis=1).values
y_train = train_set.drop(columns=['X1','X2','X3'], axis=1).values
X_test = test_set.drop(columns=['Y'], axis=1).values 
y_test = test_set.drop(columns=['X1','X2','X3'], axis=1).values


# In[4]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[5]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import optuna
from sklearn.model_selection import train_test_split, KFold


# In[6]:


from sklearn.tree import DecisionTreeRegressor
tree_model = DecisionTreeRegressor()


# In[7]:


from sklearn.model_selection import GridSearchCV
tree_model = DecisionTreeRegressor()  
param_grid = [{'max_depth': [2,3,4,5,6,7,8,9,10],
               'min_samples_split':[1,2,3,4,5,6,7,8,9],'min_samples_leaf':[1,2,3,45,6,]}]
grid_search = GridSearchCV (tree_model, param_grid,cv=5, scoring = 'neg_mean_squared_error' )
grid_search.fit(X_train,y_train)


# In[8]:


grid_search.best_params_


# In[16]:


tree_model = DecisionTreeRegressor(max_depth =7, min_samples_leaf= 1,min_samples_split= 2) 
model = tree_model.fit(X_train,y_train )
trainpre = tree_model.predict(X_train)
mse_in_train = mean_squared_error (y_train , trainpre)
rmse_in_train = np.sqrt (mse_in_train)
R2_in_train = r2_score(y_train , trainpre)
print ("training R2：", R2_in_train)


# In[14]:


tree_model = DecisionTreeRegressor(max_depth = 7, min_samples_leaf= 1,min_samples_split= 2) #加入调好的参数
model = tree_model.fit(X_train,y_train )
testpre = tree_model.predict(X_test)
mse_in_test = mean_squared_error (y_test , testpre)
rmse_in_test = np.sqrt (mse_in_test)
R2_in_test = r2_score(y_test , testpre)
print ("test R2：", R2_in_test)


# In[ ]:




