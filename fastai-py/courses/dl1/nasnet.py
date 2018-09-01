
# coding: utf-8

# # NasNet Dogs v Cats

# In[1]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from fastai.conv_learner import *
PATH = "/eee/dogscats/"
sz=224; bs=48


# In[ ]:


def nasnet(pre): return nasnetalarge(pretrained = 'imagenet' if pre else None)
model_features[nasnet]=4032*2


# In[3]:


stats = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
tfms = tfms_from_stats(stats, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
data = ImageClassifierData.from_paths(PATH, tfms=tfms, bs=bs)


# In[17]:


learn = ConvLearner.pretrained(nasnet, data, precompute=True, xtra_fc=[], ps=0.5)


# In[1]:


get_ipython().run_line_magic('time', 'learn.fit(1e-2, 2)')


# In[19]:


learn.precompute=False
learn.bn_freeze=True


# In[2]:


get_ipython().run_line_magic('time', 'learn.fit(1e-2, 1, cycle_len=1)')


# In[21]:


learn.save('nas_pre')


# In[28]:


def freeze_to(m, n):
    c=children(m[0])
    for l in c:     set_trainable(l, False)
    for l in c[n:]: set_trainable(l, True)


# In[29]:


freeze_to(learn.model, 17)


# In[3]:


learn.fit([1e-5,1e-4,1e-2], 1, cycle_len=1)


# In[9]:


learn.save('nas')

