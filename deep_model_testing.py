#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import random


# In[2]:


test_premise = np.loadtxt('test/premise.txt', delimiter=',')


# In[3]:


test_hypothesis = np.loadtxt('test/hypothesis.txt', delimiter=',')


# In[4]:


test_labels = np.loadtxt('test/labels.txt', delimiter=',')


# In[5]:


test_raw = np.subtract(test_premise,test_hypothesis)


# In[6]:


testY_raw = test_labels


# In[30]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4096,1024)
        self.fc2 = nn.Linear(1024,256)
        self.fc3 = nn.Linear(256,64)
        self.fc4 = nn.Linear(64,3)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# In[49]:


model = Net()
model.load_state_dict(torch.load('models/rajat.pt', map_location={'cuda:0': 'cpu'}))


# In[50]:


testX = torch.from_numpy(test_raw).float()
Ypred = model(testX)


# In[51]:


def idx2word(word):
    if(word == 0):
        return 'neutral';
    elif(word == 1):
        return 'entailment';
    elif(word == 2):
        return 'contradiction';


# In[52]:


Ypred = np.argmax(np.array(Ypred.cpu().detach().numpy()), axis=1)


# In[53]:


output = [idx2word(item) for item in Ypred]


# In[54]:


output[50:1000]


# In[56]:


with open('deep_model.txt', 'w') as f:
    for item in output:
        f.write("%s\n" % item)


# In[ ]:




