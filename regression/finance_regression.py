#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


# In[3]:


class StrToBytes:
    def __init__(self, fileobj):
        self.fileobj = fileobj
    def read(self, size):
        return self.fileobj.read(size).encode()
    def readline(self, size=-1):
        return self.fileobj.readline(size).encode()


# In[14]:


dictionary = pickle.load(StrToBytes(open("../final_project/final_project_dataset_modified.pkl", "r")))


# In[15]:


### list the features you want to look at--first item in the 
### list will be the "target" feature
features_list = ["bonus", "salary"]
data = featureFormat( dictionary, features_list, remove_any_zeroes=True)
data


# In[16]:


target, features = targetFeatureSplit( data )


# In[17]:


### training-testing split needed in regression, just like classification
from sklearn.model_selection import train_test_split
feature_train, feature_test, target_train, target_test =  train_test_split(features, target, test_size=0.5, random_state=42)
train_color = "b"
test_color = "r"


# In[18]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(feature_train,target_train)
print(reg.coef_)
print(reg.intercept_)
print(reg.score(feature_train,target_train))


# In[23]:


### draw the scatterplot, with color-coded training and testing points
import matplotlib.pyplot as plt
for feature, target in zip(feature_test, target_test):
    plt.scatter( feature, target, color=test_color ) 
for feature, target in zip(feature_train, target_train):
    plt.scatter( feature, target, color=train_color ) 

### labels for the legend
plt.scatter(feature_test[0], target_test[0], color=test_color, label="test")
plt.scatter(feature_test[0], target_test[0], color=train_color, label="train")



#reg.fit(feature_test, target_test)
### draw the regression line, once it's coded
try:
    plt.plot( feature_test, reg.predict(feature_test) )
except NameError:
    pass
plt.xlabel(features_list[1])
plt.ylabel(features_list[0])
plt.legend()
plt.show()


# In[15]:


print(reg.coef_)
print(reg.intercept_)


# In[16]:


print(reg.score(feature_test,target_test))


# In[ ]:




