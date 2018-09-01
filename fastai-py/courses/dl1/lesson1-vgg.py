
# coding: utf-8

# ## Image classification with Convolutional Neural Networks

# In[1]:


# Put these at the top of every notebook, to get automatic reloading and inline plotting
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# This file contains all the main external libs we'll use
from fastai.imports import *

from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *


# In[3]:


PATH = "/eee/dogscats/"
sz=224
arch=vgg16
bs=64


# In[4]:


# Uncomment the below if you need to reset your precomputed activations
# !rm -rf {PATH}tmp


# In[5]:


data = ImageClassifierData.from_paths(PATH, tfms=tfms_from_model(arch, sz))


# In[9]:


learn = ConvLearner.pretrained(arch, data, precompute=True)


# In[10]:


learn.fit(0.01, 3, cycle_len=1)


# In[4]:


tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)


# In[5]:


data = ImageClassifierData.from_paths(PATH, tfms=tfms, bs=bs, num_workers=4)
learn = ConvLearner.pretrained(arch, data, precompute=True)


# In[6]:


learn.fit(1e-2, 2)


# In[7]:


learn.precompute=False


# In[8]:


learn.fit(1e-2, 1, cycle_len=1)


# In[8]:


learn.unfreeze()


# In[9]:


lr=np.array([1e-4,1e-3,1e-2])


# In[10]:


learn.fit(lr, 1, cycle_len=1)


# In[11]:


learn.fit(lr, 3, cycle_len=1, cycle_mult=2)


# In[14]:


learn.fit(lr, 3, cycle_len=3)


# In[15]:


log_preds,y = learn.TTA()
probs = np.mean(np.exp(log_preds),0)accuracy_np(probs,y)

