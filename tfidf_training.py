#!/usr/bin/env python
# coding: utf-8

# In[41]:


import pandas as pd
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression


# In[42]:


data=pd.read_csv("data/snli_1.0/snli_1.0_train.csv",header=0)


# In[44]:


array1 = data[data.label=='neutral'].as_matrix()[:,:-1]
array2 = data[data.label=='contradiction'].as_matrix()[:,:-1]
array3 = data[data.label=='entailment'].as_matrix()[:,:-1]


# In[ ]:


y1=np.zeros((array1.shape[0],1))
y2=np.ones((array2.shape[0],1))
y3=2*np.ones((array3.shape[0],1))


# In[56]:


array=np.append(array1,array2,axis=0)
array=np.append(array,array3,axis=0)
y=np.append(y1,y2,axis=0)
y=np.append(y,y3,axis=0)


# In[6]:


vectorizer1 = TfidfVectorizer(max_df=0.99, min_df=5, use_idf=True, ngram_range=(1, 2))
X1 = vectorizer1.fit_transform(array[:,0])
vectorizer2 = TfidfVectorizer(max_df=0.99, min_df=5,use_idf=True, ngram_range=(1, 2))
X2 = vectorizer2.fit_transform(array[:,1])


# In[7]:


X=hstack([X1,X2])


# In[8]:


clf = LogisticRegression(random_state=0).fit(X, y)


# In[28]:


torch.save(clf, 'models/tfidf.pt')


# In[54]:


torch.save(vectorizer1, 'models/v1.pt')


# In[55]:


torch.save(vectorizer2, 'models/v2.pt')


# In[ ]:




