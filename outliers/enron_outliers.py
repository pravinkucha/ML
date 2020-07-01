#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import sys
import matplotlib.pyplot as plt
sys.path.append('../tools/')
from feature_format import featureFormat , targetFeatureSplit


# In[2]:


class StrToBytes:
    def __init__(self, fileobj):
        self.fileobj = fileobj
    def read(self, size):
        return self.fileobj.read(size).encode()
    def readline(self, size=-1):
        return self.fileobj.readline(size).encode()


# In[3]:


data_dict = pickle.load(open('../final_project/final_project_dataset.pkl','rb'))
features  = ['salary','bonus']
data_dict.pop('TOTAL',0)
data      = featureFormat(data_dict,features)
data_dict


# In[4]:


salary = data[:,0]
bonus  = data[:,1]
salary.shape
bonus.shape


# In[5]:


plt.scatter(salary,bonus)
plt.xlabel("Salary")
plt.ylabel("Bonus")
plt.show()


# In[6]:


salary = salary.reshape(-1,1)
bonus  = bonus.reshape(-1,1)


# In[7]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(salary,bonus)
print(reg.coef_)
print(reg.intercept_)
print(reg.score(salary,bonus))


# In[8]:


prediction = reg.predict(salary)


# In[9]:


mse = (bonus - prediction) **2
cleand_data = zip(salary,bonus,prediction)
cleand_data = sorted(cleand_data,key=lambda x : x[2][0],reverse=True)
cleand_data


# In[10]:


outlier_salary = cleand_data[0][0][0]
outlier_salary


# In[11]:


outlier_bonus = cleand_data[0][1][0]
outlier_bonus


# In[12]:


outlier_mse = cleand_data[0][2][0]
outlier_mse


# In[24]:


outliers = []
for key in data_dict:
    val = data_dict[key]['salary']
    if val == 'NaN':
        continue
    outliers.append((key,int(val)))


# In[25]:


print(sorted(outliers,key=lambda x : x[1],reverse=True)[:2])


# In[26]:


outliers


# In[ ]:




