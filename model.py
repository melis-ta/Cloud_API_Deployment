#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle


# In[2]:


#read the csv file and take a copy.
dataset = pd.read_csv(r'C:\Users\hmeli\OneDrive\Masaüstü\DataGlacierDataSets\diabetes.csv')
df=dataset.copy()


# In[3]:


#First look at the dataset
df.head()


# In[4]:


#informations about the dataset. 
df.info()


# In[5]:


#statistical informations of the data set.
df.describe().T


# ## Modelling
# 
# Creating the dependent and independent variables to apply Logistic regression model. 

# In[6]:


df["Outcome"].value_counts()


# In[7]:


X= df.iloc[:, :8]
y=df["Outcome"]


# In[8]:


#check the partition X.
X.head()


# In[9]:


#splitting the dataset as train set and test set by using sklearn.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.30, random_state = 101)
loj=LogisticRegression(solver="liblinear")


# In[10]:


#fitting the model with training set.
loj_model= loj.fit(X_train,y_train)
loj_model


# In[11]:


pickle.dump(loj, open('model.pickle','wb'))


# In[ ]:




