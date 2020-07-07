#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pickle
import json
import sys
sys.path.append('../final_project/')
from poi_email_addresses import poiEmails


enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))
len(enron_data)


# In[48]:


salry_cnt = 0;
email_cnt = 0;
for x in enron_data:
    if enron_data[x]['total_payments'] =='NaN':
       salry_cnt +=  1
    if enron_data[x]['email_address'] !='NaN':
       email_cnt +=  1


# In[49]:


print(salry_cnt)
print(email_cnt)


# In[40]:


poiEmails_list = poiEmails()
poiEmails_list


# In[43]:


len(poiEmails_list)


# In[45]:


i = 0;
lenth = 0;
for x in enron_data:
    if enron_data[x]['poi'] ==True:
        lenth += 1
        if enron_data[x]['total_payments'] =='NaN':
            i +=  1      


# In[46]:


lenth


# In[47]:


i


# In[54]:


class StrToBytes:
    def __init__(self, fileobj):
        self.fileobj = fileobj
    def read(self, size):
        return self.fileobj.read(size).encode()
    def readline(self, size=-1):
        return self.fileobj.readline(size).encode()


# In[105]:


list_poi_name = []
import re
bad_chars = ['(y)', '(n)', '\S', "\s"]
p = re.compile('|'.join(map(re.escape, to_remove))) # escape to handle metachars
with open('../final_project/poi_names.txt','r') as file_poi_name :
    for x in file_poi_name:
        if len(x) > 1 and not re.search('http', x): 
            x = re.sub("(y)|(n)|()","",x)
            x = re.sub('[^A-Za-z0-9,]+', '', x)
            list_poi_name.append(x.split(","))
            l = l + 1


# In[106]:


list_poi_name


# In[112]:


list_poi_name[0][1]


# In[132]:


sorted(enron_data.keys())


# In[133]:


enron_data['FASTOW ANDREW S']['total_payments']


# In[134]:


enron_data['LAY KENNETH L']['total_payments']


# In[135]:


enron_data['SKILLING JEFFREY K']['total_payments']


# In[31]:


import itertools
data = dict(itertools.islice(enron_data.items(), 1))
data


# In[29]:


import sys
sys.path.append('../tools/')
from feature_format import featureFormat,targetFeatureSplit
feature_list = ["salary","bonus","poi"] 
data_array = featureFormat(enron_data, feature_list)
data_array


# In[ ]:




