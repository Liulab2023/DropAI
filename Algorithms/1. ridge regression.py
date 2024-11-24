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
from sklearn.linear_model import LinearRegression, Ridge ,Lasso
from sklearn.model_selection import train_test_split, KFold


# In[5]:


ridge_model = Ridge ()


# In[6]:


from sklearn.model_selection import GridSearchCV
param_grid = [{'alpha': np.linspace(0,1,101)}]
grid_search = GridSearchCV (ridge_model, param_grid,cv=5, scoring = 'neg_mean_squared_error' )
grid_search.fit(X_train,y_train)


# In[7]:


grid_search.best_params_


# In[8]:


from sklearn.model_selection import GridSearchCV
param_grid = [{'alpha': np.linspace(0,1,101)}]
grid_search = GridSearchCV (ridge_model, param_grid,cv=5, scoring = 'neg_mean_squared_error' )
grid_search.fit(X_train,y_train)


# In[9]:


grid_search.best_params_


# In[10]:


Ridge_model = Ridge(alpha=1.0)
model = Ridge_model.fit(X_train,y_train)
trainpre = Ridge_model.predict(X_train)
mse_in_train = mean_squared_error (y_train , trainpre)
rmse_in_train = np.sqrt (mse_in_train)
R2_in_train = r2_score(y_train , trainpre)
print ("training R2：", R2_in_train)


# In[11]:


Ridge_model = Ridge(alpha=1.0)
model = Ridge_model.fit(X_train,y_train)
testpre = Ridge_model.predict(X_test)
mse_in_test = mean_squared_error (y_test , testpre)
rmse_in_test = np.sqrt (mse_in_test)
R2_in_test = r2_score(y_test , testpre)
print ("test R2：", R2_in_test)


# In[ ]:




