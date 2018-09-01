
# coding: utf-8

# ## CIFAR 10

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


from fastai.conv_learner import *
PATH = "data/cifar10/"
os.makedirs(PATH,exist_ok=True)


# In[3]:


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
stats = (np.array([ 0.4914 ,  0.48216,  0.44653]), np.array([ 0.24703,  0.24349,  0.26159]))


# In[4]:


def get_data(sz,bs):
    tfms = tfms_from_stats(stats, sz, aug_tfms=[RandomFlip()], pad=sz//8)
    return ImageClassifierData.from_paths(PATH, val_name='test', tfms=tfms, bs=bs)


# In[5]:


bs=128


# ### Look at data

# In[ ]:


data = get_data(32,4)


# In[6]:


x,y=next(iter(data.trn_dl))


# In[12]:


plt.imshow(data.trn_ds.denorm(x)[0]);


# In[13]:


plt.imshow(data.trn_ds.denorm(x)[1]);


# ## Initial model

# In[6]:


from fastai.models.cifar10.resnext import resnext29_8_64

m = resnext29_8_64()
bm = BasicModel(m.cuda(), name='cifar10_rn29_8_64')


# In[7]:


data = get_data(8,bs*4)


# In[8]:


learn = ConvLearner(data, bm)
learn.unfreeze()


# In[8]:


lr=1e-2; wd=5e-4


# In[9]:


learn.lr_find()


# In[10]:


learn.sched.plot()


# In[26]:


get_ipython().run_line_magic('time', 'learn.fit(lr, 1)')


# In[27]:


learn.fit(lr, 2, cycle_len=1)


# In[28]:


learn.fit(lr, 3, cycle_len=1, cycle_mult=2, wds=wd)


# In[30]:


learn.save('8x8_8')


# ## 16x16

# In[30]:


learn.load('8x8_8')


# In[13]:


learn.set_data(get_data(16,bs*2))


# In[14]:


get_ipython().run_line_magic('time', 'learn.fit(1e-3, 1, wds=wd)')


# In[15]:


learn.unfreeze()


# In[16]:


learn.lr_find()


# In[17]:


learn.sched.plot()


# In[18]:


lr=1e-2


# In[19]:


learn.fit(lr, 2, cycle_len=1, wds=wd)


# In[20]:


learn.fit(lr, 3, cycle_len=1, cycle_mult=2, wds=wd)


# In[21]:


learn.save('16x16_8')


# ## 24x24

# In[9]:


learn.load('16x16_8')


# In[10]:


learn.set_data(get_data(24,bs))


# In[11]:


learn.fit(1e-2, 1, wds=wd)


# In[12]:


learn.unfreeze()


# In[13]:


learn.fit(lr, 1, cycle_len=1, wds=wd)


# In[14]:


learn.fit(lr, 3, cycle_len=1, cycle_mult=2, wds=wd)


# In[15]:


learn.save('24x24_8')


# In[16]:


log_preds,y = learn.TTA()
preds = np.mean(np.exp(log_preds),0), metrics.log_loss(y,preds), accuracy_np(preds,y)


# ## 32x32

# In[9]:


learn.load('24x24_8')


# In[20]:


learn.set_data(get_data(32,bs))


# In[21]:


learn.fit(1e-2, 1, wds=wd)


# In[22]:


learn.unfreeze()


# In[23]:


learn.fit(lr, 3, cycle_len=1, cycle_mult=2, wds=wd)


# In[26]:


learn.fit(lr, 3, cycle_len=4, wds=wd)


# In[27]:


log_preds,y = learn.TTA()
metrics.log_loss(y,np.exp(log_preds)), accuracy_np(log_preds,y)


# In[ ]:


learn.save('32x32_8')

