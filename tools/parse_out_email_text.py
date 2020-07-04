#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/python

from nltk.stem.snowball import SnowballStemmer
import string
import nltk
obj_stem = SnowballStemmer('english')


# In[2]:


def parseOutText(f):
    """ given an opened email file f, parse out all text below the
        metadata block at the top
        (in Part 2, you will also add stemming capabilities)
        and return a string that contains all the words
        in the email (space-separated) 
        
        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)
        
        """


    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()

    ### split off metadata
    content = all_text.split("X-FileName:")
    words = ""
    if len(content) > 1:
        ### remove punctuation
        text_string = content[1].translate(content[1].maketrans("", "",string.punctuation))
        words  = nltk.word_tokenize(text_string)
        words  = [obj_stem.stem(word) for word in words if word]
        #print(words)
        ### project part 2: comment out the line below
        words =' '.join(words)
        

        ### split the text string into individual words, stem each word,
        ### and append the stemmed word to words (make sure there's a single
        ### space between each stemmed word)
        




    return words


# In[3]:


def main():
    ff = open("../text_learning/test_email.txt", "r")
    text = parseOutText(ff)
    print (text)


# In[4]:


if __name__ == '__main__':
    main()


# In[ ]:




