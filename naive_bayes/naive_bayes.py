#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


# In[3]:


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


# In[4]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(features_train,labels_train)


# In[5]:


pred_y = gnb.predict(features_test)
pred_y


# In[6]:


print("Number of mislabeled points out of a total %d points : %d"
      % (features_test.shape[0], (labels_test != pred_y).sum()))


# In[7]:


from sklearn.metrics import accuracy_score
scor = accuracy_score(pred_y,labels_test)
scor


# In[ ]:




