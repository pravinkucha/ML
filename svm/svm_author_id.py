#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time,ctime
from datetime import datetime
sys.path.append("../tools/")
from email_preprocess import preprocess


# In[2]:


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
print("splite start =", ctime())
t1 = datetime.now()
features_train, features_test, labels_train, labels_test = preprocess()
print("splite end =", ctime())
t2 = datetime.now()
delta = t2 - t1
print(delta.total_seconds())


# In[3]:


print("train time start =", ctime())
t1 = datetime.now()
from sklearn.svm import SVC
clf = SVC(kernel='rbf')
clf.fit(features_train,labels_train)
print("train time end =", ctime())
t2 = datetime.now()
delta = t2 - t1
print(delta.total_seconds())


# In[4]:


print("predict time start =", ctime())
t1 = datetime.now()
y_pred = clf.predict(features_test)
print("predict time end =", ctime())
t2 = datetime.now()
delta = t2 - t1
print(delta.total_seconds())
y_pred


# In[5]:


print("Number of mislabeled points out of a total %d points : %d"
      % (features_test.shape[0], (labels_test != y_pred).sum()))


# In[6]:


from sklearn.metrics import accuracy_score
scor = accuracy_score(y_pred,labels_test)
scor


# In[ ]:




