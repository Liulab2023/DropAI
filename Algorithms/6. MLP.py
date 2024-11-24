#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv(r'data.csv',parse_dates=[3])
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
X_train = train_set.drop(columns=['Y'], axis=1).values
y_train = train_set.drop(columns=['X1','X2','X3'], axis=1).values
X_test = test_set.drop(columns=['Y'], axis=1).values 
y_test = test_set.drop(columns=['X1','X2','X3'], axis=1).values


# In[3]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[4]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression,  Ridge ,Lasso
from sklearn.model_selection import train_test_split, KFold


# In[5]:


from sklearn.model_selection import GridSearchCV
MLP_model  = MLPRegressor()  
param_grid = [{'activation':['logistic','tanh','relu','‘identity'],
               'hidden_layer_sizes':[(50,50),(50,100),(100,50),(100,100)],
               'alpha':[0.001, 0.01, 0.1, 0.2, 0.4, 1, 10],
               'solver':['lbfgs','sgd','adam']
              }]
grid_search = GridSearchCV (MLP_model, param_grid,cv=5, scoring = 'neg_mean_squared_error' )
grid_search.fit(X_train,y_train)


# In[57]:


MLP_model  = MLPRegressor(**grid_search.best_params_ )
model = MLP_model.fit(X_train,y_train)
trainpre =MLP_model.predict(X_train)
R2_in_train = r2_score(y_train , trainpre)
print ("training R2：", R2_in_train)


# In[62]:


MLP_model  = MLPRegressor(**grid_search.best_params_)
model = MLP_model.fit(X_test,y_test)
testpre =MLP_model.predict(X_test)
R2_in_test = r2_score(y_test , testpre)
print ("training R2：", R2_in_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




