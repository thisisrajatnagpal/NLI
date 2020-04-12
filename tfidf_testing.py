#!/usr/bin/env python
# coding: utf-8

# In[7]:


import torch
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack


# In[3]:


model = torch.load('models/tfidf.pt')
vectorizer1 = torch.load('models/v1.pt')
vectorizer2 = torch.load('models/v2.pt')


# In[6]:


data_test=pd.read_csv("data/snli_1.0/snli_1.0_test.csv",header=0)


# In[8]:


array_test=data_test.as_matrix(('sentence1','sentence2'))
y_test=data_test.as_matrix(('label',))


# In[21]:


x1=vectorizer1.transform(array_test[:,0])
x2=vectorizer2.transform(array_test[:,1])
x=hstack([x1,x2])


# In[22]:


y_out = model.predict(x)


# In[23]:


def idx2word(word):
    if(word == 0):
        return 'neutral';
    elif(word == 1):
        return 'entailment';
    elif(word == 2):
        return 'contradiction';


# In[24]:


output = [idx2word(item) for item in y_out]


# In[ ]:


with open('tfidf.txt', 'w') as f:
    for item in output:
        f.write("%s\n" % item)

