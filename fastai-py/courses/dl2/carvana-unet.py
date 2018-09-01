
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


from fastai.conv_learner import *
from fastai.dataset import *
from fastai.models.resnet import vgg_resnet50

import json


# In[3]:


torch.cuda.set_device(2)


# In[4]:


torch.backends.cudnn.benchmark=True


# ## Data

# In[5]:


PATH = Path('data/carvana')
MASKS_FN = 'train_masks.csv'
META_FN = 'metadata.csv'
masks_csv = pd.read_csv(PATH/MASKS_FN)
meta_csv = pd.read_csv(PATH/META_FN)


# In[6]:


def show_img(im, figsize=None, ax=None, alpha=None):
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im, alpha=alpha)
    ax.set_axis_off()
    return ax


# In[7]:


TRAIN_DN = 'train-128'
MASKS_DN = 'train_masks-128'
sz = 128
bs = 64
nw = 16


# In[7]:


TRAIN_DN = 'train'
MASKS_DN = 'train_masks_png'
sz = 128
bs = 64
nw = 16


# In[8]:


class MatchedFilesDataset(FilesDataset):
    def __init__(self, fnames, y, transform, path):
        self.y=y
        assert(len(fnames)==len(y))
        super().__init__(fnames, transform, path)
    def get_y(self, i): return open_image(os.path.join(self.path, self.y[i]))
    def get_c(self): return 0


# In[9]:


x_names = np.array([Path(TRAIN_DN)/o for o in masks_csv['img']])
y_names = np.array([Path(MASKS_DN)/f'{o[:-4]}_mask.png' for o in masks_csv['img']])


# In[10]:


val_idxs = list(range(1008))
((val_x,trn_x),(val_y,trn_y)) = split_by_idx(val_idxs, x_names, y_names)


# In[11]:


aug_tfms = [RandomRotate(4, tfm_y=TfmType.CLASS),
            RandomFlip(tfm_y=TfmType.CLASS),
            RandomLighting(0.05, 0.05, tfm_y=TfmType.CLASS)]


# In[12]:


tfms = tfms_from_model(resnet34, sz, crop_type=CropType.NO, tfm_y=TfmType.CLASS, aug_tfms=aug_tfms)
datasets = ImageData.get_ds(MatchedFilesDataset, (trn_x,trn_y), (val_x,val_y), tfms, path=PATH)
md = ImageData(PATH, datasets, bs, num_workers=16, classes=None)
denorm = md.trn_ds.denorm


# In[13]:


x,y = next(iter(md.trn_dl))


# In[14]:


x.shape,y.shape


# ## Simple upsample

# In[15]:


f = resnet34
cut,lr_cut = model_meta[f]


# In[16]:


def get_base():
    layers = cut_model(f(True), cut)
    return nn.Sequential(*layers)


# In[17]:


def dice(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()


# In[18]:


class StdUpsample(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.conv = nn.ConvTranspose2d(nin, nout, 2, stride=2)
        self.bn = nn.BatchNorm2d(nout)
        
    def forward(self, x): return self.bn(F.relu(self.conv(x)))


# In[19]:


class Upsample34(nn.Module):
    def __init__(self, rn):
        super().__init__()
        self.rn = rn
        self.features = nn.Sequential(
            rn, nn.ReLU(),
            StdUpsample(512,256),
            StdUpsample(256,256),
            StdUpsample(256,256),
            StdUpsample(256,256),
            nn.ConvTranspose2d(256, 1, 2, stride=2))
        
    def forward(self,x): return self.features(x)[:,0]


# In[20]:


class UpsampleModel():
    def __init__(self,model,name='upsample'):
        self.model,self.name = model,name

    def get_layer_groups(self, precompute):
        lgs = list(split_by_idxs(children(self.model.rn), [lr_cut]))
        return lgs + [children(self.model.features)[1:]]


# In[31]:


m_base = get_base()


# In[32]:


m = to_gpu(Upsample34(m_base))
models = UpsampleModel(m)


# In[33]:


learn = ConvLearner(md, models)
learn.opt_fn=optim.Adam
learn.crit=nn.BCEWithLogitsLoss()
learn.metrics=[accuracy_thresh(0.5),dice]


# In[25]:


learn.freeze_to(1)


# In[62]:


learn.lr_find()
learn.sched.plot()


# In[21]:


lr=4e-2
wd=1e-7
lrs = np.array([lr/100,lr/10,lr])/2


# In[35]:


learn.fit(lr,1, wds=wd, cycle_len=4,use_clr=(20,8))


# In[36]:


learn.save('tmp')


# In[27]:


learn.load('tmp')


# In[ ]:


learn.unfreeze()
learn.bn_freeze(True)


# In[38]:


learn.fit(lrs,1,cycle_len=4,use_clr=(20,8))


# In[39]:


learn.save('128')


# In[40]:


x,y = next(iter(md.val_dl))
py = to_np(learn.model(V(x)))


# In[41]:


show_img(py[0]>0);


# In[42]:


show_img(y[0]);


# ## U-net (ish)

# In[22]:


class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = output
    def remove(self): self.hook.remove()


# In[23]:


class UnetBlock(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        up_out = x_out = n_out//2
        self.x_conv  = nn.Conv2d(x_in,  x_out,  1)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
        self.bn = nn.BatchNorm2d(n_out)
        
    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p,x_p], dim=1)
        return self.bn(F.relu(cat_p))


# In[24]:


class Unet34(nn.Module):
    def __init__(self, rn):
        super().__init__()
        self.rn = rn
        self.sfs = [SaveFeatures(rn[i]) for i in [2,4,5,6]]
        self.up1 = UnetBlock(512,256,256)
        self.up2 = UnetBlock(256,128,256)
        self.up3 = UnetBlock(256,64,256)
        self.up4 = UnetBlock(256,64,256)
        self.up5 = nn.ConvTranspose2d(256, 1, 2, stride=2)
        
    def forward(self,x):
        x = F.relu(self.rn(x))
        x = self.up1(x, self.sfs[3].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        x = self.up5(x)
        return x[:,0]
    
    def close(self):
        for sf in self.sfs: sf.remove()


# In[25]:


class UnetModel():
    def __init__(self,model,name='unet'):
        self.model,self.name = model,name

    def get_layer_groups(self, precompute):
        lgs = list(split_by_idxs(children(self.model.rn), [lr_cut]))
        return lgs + [children(self.model)[1:]]


# In[48]:


m_base = get_base()
m = to_gpu(Unet34(m_base))
models = UnetModel(m)


# In[49]:


learn = ConvLearner(md, models)
learn.opt_fn=optim.Adam
learn.crit=nn.BCEWithLogitsLoss()
learn.metrics=[accuracy_thresh(0.5),dice]


# In[38]:


learn.summary()


# In[39]:


[o.features.size() for o in m.sfs]


# In[50]:


learn.freeze_to(1)


# In[53]:


learn.lr_find()
learn.sched.plot()


# In[25]:


lr=4e-2
wd=1e-7

lrs = np.array([lr/100,lr/10,lr])


# In[52]:


learn.fit(lr,1,wds=wd,cycle_len=8,use_clr=(5,8))


# In[53]:


learn.save('128urn-tmp')


# In[56]:


learn.load('128urn-tmp')


# In[57]:


learn.unfreeze()
learn.bn_freeze(True)


# In[58]:


learn.fit(lrs/4, 1, wds=wd, cycle_len=20,use_clr=(20,10))


# In[59]:


learn.save('128urn-0')


# In[26]:


learn.load('128urn-0')


# In[60]:


x,y = next(iter(md.val_dl))
py = to_np(learn.model(V(x)))


# In[61]:


show_img(py[0]>0);


# In[62]:


show_img(y[0]);


# In[63]:


m.close()


# ## 512x512

# In[64]:


sz=512
bs=16


# In[65]:


tfms = tfms_from_model(resnet34, sz, crop_type=CropType.NO, tfm_y=TfmType.CLASS, aug_tfms=aug_tfms)
datasets = ImageData.get_ds(MatchedFilesDataset, (trn_x,trn_y), (val_x,val_y), tfms, path=PATH)
md = ImageData(PATH, datasets, bs, num_workers=4, classes=None)
denorm = md.trn_ds.denorm


# In[66]:


m_base = get_base()
m = to_gpu(Unet34(m_base))
models = UnetModel(m)


# In[67]:


learn = ConvLearner(md, models)
learn.opt_fn=optim.Adam
learn.crit=nn.BCEWithLogitsLoss()
learn.metrics=[accuracy_thresh(0.5),dice]


# In[68]:


learn.freeze_to(1)


# In[69]:


learn.load('128urn-0')


# In[70]:


learn.fit(lr,1,wds=wd, cycle_len=5,use_clr=(5,5))


# In[71]:


learn.save('512urn-tmp')


# In[72]:


learn.unfreeze()
learn.bn_freeze(True)


# In[38]:


learn.load('512urn-tmp')


# In[73]:


learn.fit(lrs/4,1,wds=wd, cycle_len=8,use_clr=(20,8))


# In[74]:


learn.save('512urn')


# In[26]:


learn.load('512urn')


# In[75]:


x,y = next(iter(md.val_dl))
py = to_np(learn.model(V(x)))


# In[76]:


show_img(py[0]>0);


# In[77]:


show_img(y[0]);


# In[78]:


m.close()


# ## 1024x1024

# In[26]:


sz=1024
bs=4


# In[27]:


tfms = tfms_from_model(resnet34, sz, crop_type=CropType.NO, tfm_y=TfmType.CLASS)
datasets = ImageData.get_ds(MatchedFilesDataset, (trn_x,trn_y), (val_x,val_y), tfms, path=PATH)
md = ImageData(PATH, datasets, bs, num_workers=16, classes=None)
denorm = md.trn_ds.denorm


# In[28]:


m_base = get_base()
m = to_gpu(Unet34(m_base))
models = UnetModel(m)


# In[29]:


learn = ConvLearner(md, models)
learn.opt_fn=optim.Adam
learn.crit=nn.BCEWithLogitsLoss()
learn.metrics=[accuracy_thresh(0.5),dice]


# In[30]:


learn.load('512urn')


# In[31]:


learn.freeze_to(1)


# In[32]:


learn.fit(lr,1, wds=wd, cycle_len=2,use_clr=(5,4))


# In[33]:


learn.save('1024urn-tmp')


# In[30]:


learn.load('1024urn-tmp')


# In[31]:


learn.unfreeze()
learn.bn_freeze(True)


# In[32]:


lrs = np.array([lr/200,lr/30,lr])


# In[33]:


learn.fit(lrs/10,1, wds=wd,cycle_len=4,use_clr=(20,8))


# In[33]:


learn.fit(lrs/10,1, wds=wd,cycle_len=4,use_clr=(20,8))


# In[34]:


learn.sched.plot_loss()


# In[35]:


learn.save('1024urn')


# In[26]:


learn.load('1024urn')


# In[36]:


x,y = next(iter(md.val_dl))
py = to_np(learn.model(V(x)))


# In[37]:


show_img(py[0]>0);


# In[38]:


show_img(y[0]);

