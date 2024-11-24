#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score


# In[2]:


data = pd.read_csv(r'data.csv',parse_dates=[3])
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)


# In[3]:


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


import pandas as pd

y_train = pd.Series(y_train.ravel())
y_train = pd.to_numeric(y_train, errors='coerce')


# In[6]:


knn_reg = KNeighborsRegressor(n_neighbors=5)

# 训练模型
knn_reg.fit(X_train, y_train)

# 训练集预测
train_pred = knn_reg.predict(X_train)
# 测试集预测
test_pred = knn_reg.predict(X_test)


# In[8]:


mse_train = mean_squared_error(y_train, train_pred)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_train, train_pred)

print("training R2：", r2_train)


mse_test = mean_squared_error(y_test, test_pred)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, test_pred)

print("test R2：", r2_test)


# In[ ]:





# In[ ]:




