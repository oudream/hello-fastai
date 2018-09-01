
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


from fastai.learner import *
from fastai.dataset import *


# In[51]:


X = np.array([[0.,0.], [0,1], [1,0], [1,1]])
y = np.array([0,1,1,0])
data = (X,y)


# In[52]:


md = ImageClassifierData.from_arrays('.', data, data, bs=4)


# In[53]:


learn = Learner.from_model_data(SimpleNet([2, 10, 2]), md)
learn.crit = nn.CrossEntropyLoss()
learn.opt_fn = optim.SGD


# In[54]:


learn.fit(1., 30, metrics=[accuracy])

