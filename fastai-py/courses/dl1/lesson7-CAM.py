
# coding: utf-8

# ## Dogs v Cats

# In[1]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

from fastai.imports import *

from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
import skimage.transform.resize


# In[2]:


PATH = "/eee/dogscats/"
sz = 224
arch = resnet34
bs = 64


# In[3]:


m = arch(True)


# In[4]:


m


# In[5]:


m = nn.Sequential(*children(m)[:-2], 
                  nn.Conv2d(512, 2, 3, padding=1), 
                  nn.AdaptiveAvgPool2d(1), Flatten(), 
                  nn.LogSoftmax())


# In[4]:


tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
data = ImageClassifierData.from_paths(PATH, tfms=tfms, bs=bs)


# In[5]:


learn = ConvLearner.from_model_data(m, data)


# In[8]:


learn.freeze_to(-4)


# In[9]:


m[-1].trainable


# In[10]:


m[-4].trainable


# In[11]:


learn.fit(0.01, 1)


# In[12]:


learn.fit(0.01, 1, cycle_len=1)


# ## Class Activation Maps (CAM)

# In[6]:


class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = output
    def remove(self): self.hook.remove()


# In[7]:


x,y = next(iter(data.val_dl))


# In[10]:


x,y = x[None,1], y[None,1]
vx = Variable(x.cuda(), requires_grad=True)


# In[39]:


dx = data.val_ds.denorm(x)[0]
plt.imshow(dx);


# In[15]:


sfs = [SaveFeatures(o) for o in [m[-7], m[-6], m[-5], m[-4]]]


# In[18]:


get_ipython().run_line_magic('time', 'py = m(Variable(x.cuda()))')


# In[19]:


for o in sfs: o.remove()


# In[20]:


[o.features.size() for o in sfs]


# In[15]:


py = np.exp(to_np(py)[0]); py


# In[16]:


feat = np.maximum(0,to_np(sfs[3].features[0]))
feat.shape


# In[23]:


f2=np.dot(np.rollaxis(feat,0,3), py)
f2-=f2.min()
f2/=f2.max()
f2


# In[22]:


plt.imshow(dx)
plt.imshow(skimage.transform.resize(f2, dx.shape), alpha=0.5, cmap='hot');


# ## Model

# In[38]:


learn.unfreeze()
learn.bn_freeze(True)


# In[39]:


# 12 layer groups call for 12 lrs
lr=np.array([[1e-6]*4,[1e-4]*4,[1e-2]*4]).flatten()


# In[40]:


learn.fit(lr, 2, cycle_len=1)


# In[41]:


log_preds,y = learn.TTA()
preds = np.mean(np.exp(log_preds),0)
accuracy_np(preds,y)


# In[42]:


learn.fit(lr, 2, cycle_len=1)


# In[43]:


log_preds,y = learn.TTA()
preds = np.mean(np.exp(log_preds),0)
accuracy_np(preds,y)

