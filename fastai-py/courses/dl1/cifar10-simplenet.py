
# coding: utf-8

# ## CIFAR 10

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[ ]:


from pathlib import Path


# In[ ]:


from fastai.conv_learner import *
PATH = Path("data/cifar10/")


# In[ ]:


bs=64
sz=32


# In[ ]:


tfms = tfms_from_model(resnet18, sz, aug_tfms=[RandomFlip()], pad=sz//8)
data = ImageClassifierData.from_csv(PATH, 'train', PATH/'train.csv', tfms=tfms, bs=bs)


# In[ ]:


learn = ConvLearner.pretrained(resnet18, data)


# In[ ]:


lr=1e-2; wd=1e-5


# In[ ]:


learn.lr_find()
learn.sched.plot()


# In[ ]:


learn.fit(lr, 1, cycle_len=1)


# In[ ]:


lrs = np.array([lr/9,lr/3,lr])


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find(lrs/1000)
learn.sched.plot()


# In[ ]:


learn.fit(lrs, 1, cycle_len=1, wds=wd)


# ## Simplenet

# In[ ]:


stats = (np.array([ 0.4914 ,  0.48216,  0.44653]), np.array([ 0.24703,  0.24349,  0.26159]))


# In[ ]:


tfms = tfms_from_stats(stats, sz, aug_tfms=[RandomFlip()], pad=sz//8)
data = ImageClassifierData.from_csv(PATH, 'train', PATH/'train.csv', tfms=tfms, bs=bs)


# In[ ]:


class SimpleConv(nn.Module):
    def __init__(self, ic, oc, ks=3, drop=0.2, bn=True):
        super().__init__()
        self.conv = nn.Conv2d(ic, oc, ks, padding=(ks-1)//2)
        self.bn = nn.BatchNorm2d(oc, momentum=0.05) if bn else None
        self.drop = nn.Dropout(drop, inplace=True)
        self.act = nn.ReLU(True)
        
    def forward(self, x):
        x = self.conv(x)
        if self.bn: x = self.bn(x)
        return self.drop(self.act(x))


# In[ ]:


net = nn.Sequential(
    SimpleConv(3, 64),
    SimpleConv(64, 128),
    SimpleConv(128, 128),
    SimpleConv(128, 128),
    nn.MaxPool2d(2),
    SimpleConv(128, 128),
    SimpleConv(128, 128),
    SimpleConv(128, 256),
    nn.MaxPool2d(2),
    SimpleConv(256, 256),
    SimpleConv(256, 256),
    nn.MaxPool2d(2),
    SimpleConv(256, 512),
    SimpleConv(512, 2048, ks=1, bn=False),
    SimpleConv(2048, 256, ks=1, bn=False),
    nn.MaxPool2d(2),
    SimpleConv(256, 256, bn=False, drop=0),
    nn.MaxPool2d(2),
    Flatten(),
    nn.Linear(256, 10)
)


# In[ ]:


bm = BasicModel(net.cuda(), name='simplenet')
learn = ConvLearner(data, bm)
learn.crit = nn.CrossEntropyLoss()
learn.opt_fn = optim.Adam
learn.unfreeze()
learn.metrics=[accuracy]
lr = 1e-3
wd = 5e-3


# In[ ]:


#sgd mom
learn.lr_find()
learn.sched.plot()


# In[ ]:


#adam
learn.lr_find()
learn.sched.plot()


# In[ ]:


learn.fit(lr, 1, wds=wd, cycle_len=20, use_clr=(32,10))


# In[ ]:


learn.fit(lr, 1, wds=wd, cycle_len=5, use_clr=(32,10))


# In[ ]:


learn.save('0')


# In[ ]:


learn.fit(lr, 3, cycle_len=1, cycle_mult=2, wds=wd)


# In[ ]:


learn.save('1')


# In[ ]:


learn.fit(lr, 1, wds=wd, cycle_len=10, use_clr=(32,10))


# In[ ]:


learn.save('2')


# ## Fin
