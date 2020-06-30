#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from datetime import datetime


# In[2]:


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
t1 = datetime.now()
features_train, features_test, labels_train, labels_test = preprocess()
t2 = datetime.now()
delta = t2 - t1
print(delta.total_seconds())
features_train.shape


# In[ ]:


t1 = datetime.now()
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(min_samples_split=40)
clf.fit(features_train,labels_train)
t2 = datetime.now()
delta = t2 - t1
print(delta.total_seconds())


# In[4]:


t1 = datetime.now()
y_pred = clf.predict(features_test)
t2 = datetime.now()
delta = t2 - t1
print(delta.total_seconds())


# In[5]:


print("Number of mislabeled points out of a total %d points : %d"
      % (features_test.shape[0], (labels_test != y_pred).sum()))


# In[6]:


from sklearn.metrics import accuracy_score
scor = accuracy_score(y_pred,labels_test)
scor


# In[7]:


print(len(features_train[0]))


# In[8]:


features_train


# In[ ]:




