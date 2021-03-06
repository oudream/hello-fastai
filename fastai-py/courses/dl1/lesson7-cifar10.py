
# coding: utf-8

# ## CIFAR 10

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# You can get the data via:
# 
#     wget http://pjreddie.com/media/files/cifar.tgz    
# **Important:** Before proceeding, the student must reorganize the downloaded dataset files to match the expected directory structure, so that there is a dedicated folder for each class under 'test' and 'train', e.g.:
# 
# ```
# * test/airplane/airplane-1001.png
# * test/bird/bird-1043.png
# 
# * train/bird/bird-10018.png
# * train/automobile/automobile-10000.png
# ```
# 
# The filename of the image doesn't have to include its class.

# In[2]:


from fastai.conv_learner import *
PATH = "data/cifar10/"
os.makedirs(PATH,exist_ok=True)

get_ipython().system('ls {PATH}')

if not os.path.exists(f"{PATH}/train/bird"):
   raise Exception("expecting class subdirs under 'train/' and 'test/'")
get_ipython().system('ls {PATH}/train')


# In[3]:


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
stats = (np.array([ 0.4914 ,  0.48216,  0.44653]), np.array([ 0.24703,  0.24349,  0.26159]))


# In[4]:


def get_data(sz,bs):
    tfms = tfms_from_stats(stats, sz, aug_tfms=[RandomFlip()], pad=sz//8)
    return ImageClassifierData.from_paths(PATH, val_name='test', tfms=tfms, bs=bs)


# In[5]:


bs=256


# ### Look at data

# In[6]:


data = get_data(32,4)


# In[7]:


x,y=next(iter(data.trn_dl))


# In[12]:


plt.imshow(data.trn_ds.denorm(x)[0]);


# In[13]:


plt.imshow(data.trn_ds.denorm(x)[1]);


# ## Fully connected model

# In[6]:


data = get_data(32,bs)


# In[7]:


lr=1e-2


# From [this notebook](https://github.com/KeremTurgutlu/deeplearning/blob/master/Exploring%20Optimizers.ipynb) by our student Kerem Turgutlu:

# In[8]:


class SimpleNet(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        for l in self.layers:
            l_x = l(x)
            x = F.relu(l_x)
        return F.log_softmax(l_x, dim=-1)


# In[9]:


learn = ConvLearner.from_model_data(SimpleNet([32*32*3, 40,10]), data)


# In[10]:


learn, [o.numel() for o in learn.model.parameters()]


# In[11]:


learn.summary()


# In[14]:


learn.lr_find()


# In[30]:


learn.sched.plot()


# In[10]:


get_ipython().run_line_magic('time', 'learn.fit(lr, 2)')


# In[11]:


get_ipython().run_line_magic('time', 'learn.fit(lr, 2, cycle_len=1)')


# ## CNN

# In[12]:


class ConvNet(nn.Module):
    def __init__(self, layers, c):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(layers[i], layers[i + 1], kernel_size=3, stride=2)
            for i in range(len(layers) - 1)])
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.out = nn.Linear(layers[-1], c)
        
    def forward(self, x):
        for l in self.layers: x = F.relu(l(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return F.log_softmax(self.out(x), dim=-1)


# In[13]:


learn = ConvLearner.from_model_data(ConvNet([3, 20, 40, 80], 10), data)


# In[14]:


learn.summary()


# In[20]:


learn.lr_find(end_lr=100)


# In[21]:


learn.sched.plot()


# In[15]:


get_ipython().run_line_magic('time', 'learn.fit(1e-1, 2)')


# In[16]:


get_ipython().run_line_magic('time', 'learn.fit(1e-1, 4, cycle_len=1)')


# ## Refactored

# In[23]:


class ConvLayer(nn.Module):
    def __init__(self, ni, nf):
        super().__init__()
        self.conv = nn.Conv2d(ni, nf, kernel_size=3, stride=2, padding=1)
        
    def forward(self, x): return F.relu(self.conv(x))


# In[45]:


class ConvNet2(nn.Module):
    def __init__(self, layers, c):
        super().__init__()
        self.layers = nn.ModuleList([ConvLayer(layers[i], layers[i + 1])
            for i in range(len(layers) - 1)])
        self.out = nn.Linear(layers[-1], c)
        
    def forward(self, x):
        for l in self.layers: x = l(x)
        x = F.adaptive_max_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return F.log_softmax(self.out(x), dim=-1)


# In[46]:


learn = ConvLearner.from_model_data(ConvNet2([3, 20, 40, 80], 10), data)


# In[47]:


learn.summary()


# In[48]:


get_ipython().run_line_magic('time', 'learn.fit(1e-1, 2)')


# In[49]:


get_ipython().run_line_magic('time', 'learn.fit(1e-1, 2, cycle_len=1)')


# ## BatchNorm

# In[17]:


class BnLayer(nn.Module):
    def __init__(self, ni, nf, stride=2, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(ni, nf, kernel_size=kernel_size, stride=stride,
                              bias=False, padding=1)
        self.a = nn.Parameter(torch.zeros(nf,1,1))
        self.m = nn.Parameter(torch.ones(nf,1,1))
        
    def forward(self, x):
        x = F.relu(self.conv(x))
        x_chan = x.transpose(0,1).contiguous().view(x.size(1), -1)
        if self.training:
            self.means = x_chan.mean(1)[:,None,None]
            self.stds  = x_chan.std (1)[:,None,None]
        return (x-self.means) / self.stds *self.m + self.a


# In[18]:


class ConvBnNet(nn.Module):
    def __init__(self, layers, c):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5, stride=1, padding=2)
        self.layers = nn.ModuleList([BnLayer(layers[i], layers[i + 1])
            for i in range(len(layers) - 1)])
        self.out = nn.Linear(layers[-1], c)
        
    def forward(self, x):
        x = self.conv1(x)
        for l in self.layers: x = l(x)
        x = F.adaptive_max_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return F.log_softmax(self.out(x), dim=-1)


# In[20]:


learn = ConvLearner.from_model_data(ConvBnNet([10, 20, 40, 80, 160], 10), data)


# In[21]:


learn.summary()


# In[22]:


get_ipython().run_line_magic('time', 'learn.fit(3e-2, 2)')


# In[23]:


get_ipython().run_line_magic('time', 'learn.fit(1e-1, 4, cycle_len=1)')


# ## Deep BatchNorm

# In[47]:


class ConvBnNet2(nn.Module):
    def __init__(self, layers, c):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5, stride=1, padding=2)
        self.layers = nn.ModuleList([BnLayer(layers[i], layers[i+1])
            for i in range(len(layers) - 1)])
        self.layers2 = nn.ModuleList([BnLayer(layers[i+1], layers[i + 1], 1)
            for i in range(len(layers) - 1)])
        self.out = nn.Linear(layers[-1], c)
        
    def forward(self, x):
        x = self.conv1(x)
        for l,l2 in zip(self.layers, self.layers2):
            x = l(x)
            x = l2(x)
        x = F.adaptive_max_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return F.log_softmax(self.out(x), dim=-1)


# In[48]:


learn = ConvLearner.from_model_data(ConvBnNet2([10, 20, 40, 80, 160], 10), data)


# In[49]:


get_ipython().run_line_magic('time', 'learn.fit(1e-2, 2)')


# In[50]:


get_ipython().run_line_magic('time', 'learn.fit(1e-2, 2, cycle_len=1)')


# ## Resnet

# In[53]:


class ResnetLayer(BnLayer):
    def forward(self, x): return x + super().forward(x)


# In[54]:


class Resnet(nn.Module):
    def __init__(self, layers, c):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5, stride=1, padding=2)
        self.layers = nn.ModuleList([BnLayer(layers[i], layers[i+1])
            for i in range(len(layers) - 1)])
        self.layers2 = nn.ModuleList([ResnetLayer(layers[i+1], layers[i + 1], 1)
            for i in range(len(layers) - 1)])
        self.layers3 = nn.ModuleList([ResnetLayer(layers[i+1], layers[i + 1], 1)
            for i in range(len(layers) - 1)])
        self.out = nn.Linear(layers[-1], c)
        
    def forward(self, x):
        x = self.conv1(x)
        for l,l2,l3 in zip(self.layers, self.layers2, self.layers3):
            x = l3(l2(l(x)))
        x = F.adaptive_max_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return F.log_softmax(self.out(x), dim=-1)


# In[55]:


learn = ConvLearner.from_model_data(Resnet([10, 20, 40, 80, 160], 10), data)


# In[56]:


wd=1e-5


# In[57]:


get_ipython().run_line_magic('time', 'learn.fit(1e-2, 2, wds=wd)')


# In[58]:


get_ipython().run_line_magic('time', 'learn.fit(1e-2, 3, cycle_len=1, cycle_mult=2, wds=wd)')


# In[59]:


get_ipython().run_line_magic('time', 'learn.fit(1e-2, 8, cycle_len=4, wds=wd)')


# ## Resnet 2

# In[63]:


class Resnet2(nn.Module):
    def __init__(self, layers, c, p=0.5):
        super().__init__()
        self.conv1 = BnLayer(3, 16, stride=1, kernel_size=7)
        self.layers = nn.ModuleList([BnLayer(layers[i], layers[i+1])
            for i in range(len(layers) - 1)])
        self.layers2 = nn.ModuleList([ResnetLayer(layers[i+1], layers[i + 1], 1)
            for i in range(len(layers) - 1)])
        self.layers3 = nn.ModuleList([ResnetLayer(layers[i+1], layers[i + 1], 1)
            for i in range(len(layers) - 1)])
        self.out = nn.Linear(layers[-1], c)
        self.drop = nn.Dropout(p)
        
    def forward(self, x):
        x = self.conv1(x)
        for l,l2,l3 in zip(self.layers, self.layers2, self.layers3):
            x = l3(l2(l(x)))
        x = F.adaptive_max_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.drop(x)
        return F.log_softmax(self.out(x), dim=-1)


# In[70]:


learn = ConvLearner.from_model_data(Resnet2([16, 32, 64, 128, 256], 10, 0.2), data)


# In[71]:


wd=1e-6


# In[72]:


get_ipython().run_line_magic('time', 'learn.fit(1e-2, 2, wds=wd)')


# In[73]:


get_ipython().run_line_magic('time', 'learn.fit(1e-2, 3, cycle_len=1, cycle_mult=2, wds=wd)')


# In[74]:


get_ipython().run_line_magic('time', 'learn.fit(1e-2, 8, cycle_len=4, wds=wd)')


# In[75]:


learn.save('tmp3')


# In[76]:


log_preds,y = learn.TTA()
preds = np.mean(np.exp(log_preds),0)


# In[77]:


metrics.log_loss(y,preds), accuracy_np(preds,y)


# ### End
