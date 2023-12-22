#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')


# In[19]:


df


# In[20]:


df.info()


# In[11]:


df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')


# In[14]:


df.describe()


# In[17]:


df.dropna(inplace=True)
df.drop('customerID', axis=1, inplace=True)


# In[18]:


df['Churn'] = df['Churn'].replace('Yes',1)
df['Churn'] = df['Churn'].replace('No',0)


# In[21]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[24]:


sns.countplot(x='Churn', data=df)


# In[26]:


for i, col in enumerate(['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']):
    plt.figure(i)
    sns.countplot(x=col, hue='Churn', data=df)


# In[30]:


X = df.drop('Churn', axis=1)
y = df['Churn']
X = pd.get_dummies(X, drop_first=True)


# In[32]:


from sklearn.model_selection import train_test_split


# In[33]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[37]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[38]:


models = [
    ('Decision Tree', DecisionTreeClassifier()),
    ('Random Forest', RandomForestClassifier(n_estimators=100)),
    ('XGBoost', XGBClassifier())]


# In[41]:


for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(name + ' accuracy: ', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))


# In[ ]:




