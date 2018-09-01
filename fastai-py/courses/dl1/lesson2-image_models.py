
# coding: utf-8

# ## Multi-label classification

# In[1]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from fastai.conv_learner import *


# In[3]:


PATH = 'data/planet/'


# In[4]:


# Data preparation steps if you are using Crestle:

os.makedirs('data/planet/models', exist_ok=True)
os.makedirs('/cache/planet/tmp', exist_ok=True)

get_ipython().system('ln -s /datasets/kaggle/planet-understanding-the-amazon-from-space/train-jpg {PATH}')
get_ipython().system('ln -s /datasets/kaggle/planet-understanding-the-amazon-from-space/test-jpg {PATH}')
get_ipython().system('ln -s /datasets/kaggle/planet-understanding-the-amazon-from-space/train_v2.csv {PATH}')
get_ipython().system('ln -s /cache/planet/tmp {PATH}')


# In[4]:


ls {PATH}


# ## Multi-label versus single-label classification

# In[5]:


from fastai.plots import *


# In[6]:


def get_1st(path): return glob(f'{path}/*.*')[0]


# In[6]:


dc_path = "/eee/dogscats/valid/"
list_paths = [get_1st(f"{dc_path}cats"), get_1st(f"{dc_path}dogs")]
plots_from_files(list_paths, titles=["cat", "dog"], maintitle="Single-label classification")


# In single-label classification each sample belongs to one class. In the previous example, each image is either a *dog* or a *cat*.

# In[7]:


list_paths = [f"{PATH}train-jpg/train_0.jpg", f"{PATH}train-jpg/train_1.jpg"]
titles=["haze primary", "agriculture clear primary water"]
plots_from_files(list_paths, titles=titles, maintitle="Multi-label classification")


# In multi-label classification each sample can belong to one or more clases. In the previous example, the first images belongs to two clases: *haze* and *primary*. The second image belongs to four clases: *agriculture*, *clear*, *primary* and  *water*.

# ## Multi-label models for Planet dataset

# In[5]:


from planet import f2

metrics=[f2]
f_model = resnet34


# In[6]:


label_csv = f'{PATH}train_v2.csv'
n = len(list(open(label_csv)))-1
val_idxs = get_cv_idxs(n)


# We use a different set of data augmentations for this dataset - we also allow vertical flips, since we don't expect vertical orientation of satellite images to change our classifications.

# In[7]:


def get_data(sz):
    tfms = tfms_from_model(f_model, sz, aug_tfms=transforms_top_down, max_zoom=1.05)
    return ImageClassifierData.from_csv(PATH, 'train-jpg', label_csv, tfms=tfms,
                    suffix='.jpg', val_idxs=val_idxs, test_name='test-jpg')


# In[9]:


data = get_data(256)


# In[35]:


x,y = next(iter(data.val_dl))


# In[36]:


y


# In[37]:


list(zip(data.classes, y[0]))


# In[43]:


plt.imshow(data.val_ds.denorm(to_np(x))[0]*1.4);


# In[8]:


sz=64


# In[9]:


data = get_data(sz)


# In[10]:


data = data.resize(int(sz*1.3), 'tmp')


# In[11]:


learn = ConvLearner.pretrained(f_model, data, metrics=metrics)


# In[15]:


lrf=learn.lr_find()
learn.sched.plot()


# In[12]:


lr = 0.2


# In[13]:


learn.fit(lr, 3, cycle_len=1, cycle_mult=2)


# In[14]:


lrs = np.array([lr/9,lr/3,lr])


# In[15]:


learn.unfreeze()
learn.fit(lrs, 3, cycle_len=1, cycle_mult=2)


# In[19]:


learn.save(f'{sz}')


# In[22]:


learn.sched.plot_loss()


# In[20]:


sz=128


# In[21]:


learn.set_data(get_data(sz))
learn.freeze()
learn.fit(lr, 3, cycle_len=1, cycle_mult=2)


# In[22]:


learn.unfreeze()
learn.fit(lrs, 3, cycle_len=1, cycle_mult=2)
learn.save(f'{sz}')


# In[23]:


sz=256


# In[24]:


learn.set_data(get_data(sz))
learn.freeze()
learn.fit(lr, 3, cycle_len=1, cycle_mult=2)


# In[25]:


learn.unfreeze()
learn.fit(lrs, 3, cycle_len=1, cycle_mult=2)
learn.save(f'{sz}')


# In[26]:


multi_preds, y = learn.TTA()
preds = np.mean(multi_preds, 0)


# In[27]:


f2(preds,y)


# ### End
