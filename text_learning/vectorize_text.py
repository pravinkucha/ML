#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pickle
import re
import sys
sys.path.append('../tools/')
from parse_out_email_text import parseOutText


# In[2]:


from_sara  = open("from_sara.txt", "r")
from_chris = open("from_chris.txt", "r")

from_data = []
word_data = []


# In[3]:


from_chris


# In[4]:


class StrToBytes:
    def __init__(self, fileobj):
        self.fileobj = fileobj
    def read(self, size):
        return self.fileobj.read(size).encode()
    def readline(self, size=-1):
        return self.fileobj.readline(size).encode()


# In[5]:


### temp_counter is a way to speed up the development--there are
### thousands of emails from Sara and Chris, so running over all of them
### can take a long time
### temp_counter helps you only look at the first 200 emails in the list so you
### can iterate your modifications quicker
temp_counter = 0
for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
    for path in from_person:
        ### only look at first 200 emails when developing
        ### once everything is working, remove this line to run over full dataset
        temp_counter += 1
        #if temp_counter < 200:
        path = os.path.join('../enron_mail_20150507/', path[:-1])
        #print (path)
        email = open(path, "r")
        words_email = parseOutText(email)
        stopwords   = {"sara", "shackleton", "chris", "germani"}
        words_email = [word for word in re.split("\W+",words_email) if word.lower() not in stopwords]
        word_data.append(' '.join(words_email))
        if name == 'sara':
            from_data.append(0)
        if name == 'chris':
            from_data.append(1)
        ### use parseOutText to extract the text from the opened email

        ### use str.replace() to remove any instances of the words
        ### ["sara", "shackleton", "chris", "germani"]

        ### append the text to word_data

        ### append a 0 to from_data if email is from Sara, and 1 if email is from Chris


        email.close()

print ("emails processed")
print(word_data)
pickle.dump( word_data, open("your_word_data.pkl", "wb"))
pickle.dump( from_data, open("your_email_authors.pkl", "wb"))


# In[6]:


from_sara.close()
from_chris.close()


# In[7]:


print(word_data[152])


# In[17]:


print (len(word_data))


# In[32]:


from nltk.corpus import stopwords
sw = stopwords.words("english")


# In[38]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english',lowercase=True)


# In[39]:


vectorizer.fit_transform(word_data)


# In[40]:


print (len(vectorizer.get_feature_names()))


# In[41]:


print (vectorizer.get_feature_names()[34597])


# In[ ]:




