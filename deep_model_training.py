#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import random


# In[3]:


train_premise = np.loadtxt('training/premise.txt', delimiter=',')


# In[4]:


train_hypothesis = np.loadtxt('training/hypothesis.txt', delimiter=',')


# In[5]:


train_labels = np.loadtxt('training/labels.txt', delimiter=',')


# In[6]:


trainX_raw = np.subtract(train_premise,train_hypothesis)


# In[11]:


trainY_raw = train_labels


# In[31]:


trainY_raw.shape[0]


# In[13]:


trainX_raw.shape


# In[81]:


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


# In[82]:


model = Net()
model.cuda()


# In[85]:


optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()


# In[86]:


for epoch in range(30):
    index = 0
    for batch_size in [64]: 
        for itera in range(17000):
            if(index > trainY_raw.shape[0]-32): break
            optimizer.zero_grad()
            trainX = torch.from_numpy(trainX_raw[index:index+batch_size]).float().cuda()
            trainY = torch.from_numpy(trainY_raw[index:index+batch_size]).long().cuda()
            Ypred = model(trainX)
            loss = criterion(Ypred, trainY)
            loss.backward()
            optimizer.step()
            index = index+batch_size
    print('epoch: ', epoch, ' loss: ', loss.item())


# In[87]:


torch.save(model.state_dict(), 'models/rajat.pt')


# In[ ]:




