
# coding: utf-8

# In[1]:


from bcolz_array_iterator2 import BcolzArrayIterator2 


# In[2]:


from bcolz import carray


# In[3]:


import numpy as np


# In[4]:


x = np.arange(14); x


# In[5]:


y = np.arange(14); y


# In[16]:


x = carray(x, chunklen=3)
y = carray(y, chunklen=3)


# In[17]:


b = BcolzArrayIterator2(x, y, shuffle=True, batch_size=3)


# In[18]:


b.N


# In[19]:


nit = len(x)//b.batch_size+1; nit


# In[20]:


for j in range(10000):
    bx,by = list(zip(*[next(b) for i in range(nit)]))
    nx = np.concatenate(bx)
    ny = np.concatenate(by)
    assert(np.allclose(nx,ny))
    assert(len(np.unique(nx))==len(nx))


# In[21]:


[next(b) for i in range(20)]

