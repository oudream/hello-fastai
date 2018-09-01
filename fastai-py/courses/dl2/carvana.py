
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


from fastai.conv_learner import *
from fastai.dataset import *

from pathlib import Path
import json
torch.cuda.set_device(1)


# ## Data

# ### Setup

# In[3]:


PATH = Path('data/carvana')
list(PATH.iterdir())


# In[4]:


MASKS_FN = 'train_masks.csv'
META_FN = 'metadata.csv'
TRAIN_DN = 'train'
MASKS_DN = 'train_masks'


# In[5]:


masks_csv = pd.read_csv(PATH/MASKS_FN)
masks_csv.head()


# In[6]:


meta_csv = pd.read_csv(PATH/META_FN)
meta_csv.head()


# In[7]:


def show_img(im, figsize=None, ax=None, alpha=None):
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im, alpha=alpha)
    ax.set_axis_off()
    return ax


# In[8]:


CAR_ID = '00087a6bd4dc'


# In[9]:


list((PATH/TRAIN_DN).iterdir())[:5]


# In[10]:


Image.open(PATH/TRAIN_DN/f'{CAR_ID}_01.jpg').resize((300,200))


# In[11]:


list((PATH/MASKS_DN).iterdir())[:5]


# In[12]:


Image.open(PATH/MASKS_DN/f'{CAR_ID}_01_mask.gif').resize((300,200))


# In[13]:


ims = [open_image(PATH/TRAIN_DN/f'{CAR_ID}_{i+1:02d}.jpg') for i in range(16)]


# In[14]:


fig, axes = plt.subplots(4, 4, figsize=(9, 6))
for i,ax in enumerate(axes.flat): show_img(ims[i], ax=ax)
plt.tight_layout(pad=0.1)


# ### Resize and convert

# In[13]:


(PATH/'train_masks_png').mkdir(exist_ok=True)


# In[14]:


def convert_img(fn):
    fn = fn.name
    Image.open(PATH/'train_masks'/fn).save(PATH/'train_masks_png'/f'{fn[:-4]}.png')


# In[23]:


files = list((PATH/'train_masks').iterdir())
with ThreadPoolExecutor(8) as e: e.map(convert_img, files)


# In[33]:


(PATH/'train_masks-128').mkdir(exist_ok=True)


# In[38]:


def resize_mask(fn):
    Image.open(fn).resize((128,128)).save((fn.parent.parent)/'train_masks-128'/fn.name)

files = list((PATH/'train_masks_png').iterdir())
with ThreadPoolExecutor(8) as e: e.map(resize_mask, files)


# In[44]:


(PATH/'train-128').mkdir(exist_ok=True)


# In[45]:


def resize_img(fn):
    Image.open(fn).resize((128,128)).save((fn.parent.parent)/'train-128'/fn.name)

files = list((PATH/'train').iterdir())
with ThreadPoolExecutor(8) as e: e.map(resize_img, files)


# ## Dataset

# In[8]:


TRAIN_DN = 'train-128'
MASKS_DN = 'train_masks-128'
sz = 128
bs = 64


# In[10]:


ims = [open_image(PATH/TRAIN_DN/f'{CAR_ID}_{i+1:02d}.jpg') for i in range(16)]
im_masks = [open_image(PATH/MASKS_DN/f'{CAR_ID}_{i+1:02d}_mask.png') for i in range(16)]


# In[11]:


fig, axes = plt.subplots(4, 4, figsize=(9, 6))
for i,ax in enumerate(axes.flat):
    ax = show_img(ims[i], ax=ax)
    show_img(im_masks[i][...,0], ax=ax, alpha=0.5)
plt.tight_layout(pad=0.1)


# In[9]:


class MatchedFilesDataset(FilesDataset):
    def __init__(self, fnames, y, transform, path):
        self.y=y
        assert(len(fnames)==len(y))
        super().__init__(fnames, transform, path)
    def get_y(self, i): return open_image(os.path.join(self.path, self.y[i]))
    def get_c(self): return 0


# In[10]:


x_names = np.array([Path(TRAIN_DN)/o for o in masks_csv['img']])
y_names = np.array([Path(MASKS_DN)/f'{o[:-4]}_mask.png' for o in masks_csv['img']])


# In[11]:


len(x_names)//16//5*16


# In[12]:


val_idxs = list(range(1008))
((val_x,trn_x),(val_y,trn_y)) = split_by_idx(val_idxs, x_names, y_names)
len(val_x),len(trn_x)


# In[13]:


aug_tfms = [RandomRotate(4, tfm_y=TfmType.CLASS),
            RandomFlip(tfm_y=TfmType.CLASS),
            RandomLighting(0.05, 0.05)]
# aug_tfms = []


# In[14]:


tfms = tfms_from_model(resnet34, sz, crop_type=CropType.NO, tfm_y=TfmType.CLASS, aug_tfms=aug_tfms)
datasets = ImageData.get_ds(MatchedFilesDataset, (trn_x,trn_y), (val_x,val_y), tfms, path=PATH)
md = ImageData(PATH, datasets, bs, num_workers=8, classes=None)


# In[57]:


denorm = md.trn_ds.denorm
x,y = next(iter(md.aug_dl))
x = denorm(x)


# In[59]:


fig, axes = plt.subplots(5, 6, figsize=(12, 10))
for i,ax in enumerate(axes.flat):
    ax=show_img(x[i], ax=ax)
    show_img(y[i], ax=ax, alpha=0.5)
plt.tight_layout(pad=0.1)


# ## Model

# In[155]:


class Empty(nn.Module): 
    def forward(self,x): return x

models = ConvnetBuilder(resnet34, 0, 0, 0, custom_head=Empty())
learn = ConvLearner(md, models)
learn.summary()


# In[15]:


class StdUpsample(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.conv = nn.ConvTranspose2d(nin, nout, 2, stride=2)
        self.bn = nn.BatchNorm2d(nout)
        
    def forward(self, x): return self.bn(F.relu(self.conv(x)))


# In[16]:


flatten_channel = Lambda(lambda x: x[:,0])


# In[17]:


simple_up = nn.Sequential(
    nn.ReLU(),
    StdUpsample(512,256),
    StdUpsample(256,256),
    StdUpsample(256,256),
    StdUpsample(256,256),
    nn.ConvTranspose2d(256, 1, 2, stride=2),
    flatten_channel
)


# In[74]:


models = ConvnetBuilder(resnet34, 0, 0, 0, custom_head=simple_up)
learn = ConvLearner(md, models)
learn.opt_fn=optim.Adam
learn.crit=nn.BCEWithLogitsLoss()
learn.metrics=[accuracy_thresh(0.5)]


# In[82]:


learn.lr_find()
learn.sched.plot()


# In[18]:


lr=4e-2


# In[76]:


learn.fit(lr,1,cycle_len=5,use_clr=(20,5))


# In[77]:


learn.save('tmp')


# In[47]:


learn.load('tmp')


# In[78]:


py,ay = learn.predict_with_targs()


# In[79]:


ay.shape


# In[80]:


show_img(ay[0]);


# In[82]:


show_img(py[0]>0);


# In[83]:


learn.unfreeze()


# In[84]:


learn.bn_freeze(True)


# In[85]:


lrs = np.array([lr/100,lr/10,lr])/4


# In[86]:


learn.fit(lrs,1,cycle_len=20,use_clr=(20,10))


# In[87]:


learn.save('0')


# In[88]:


x,y = next(iter(md.val_dl))
py = to_np(learn.model(V(x)))


# In[89]:


ax = show_img(denorm(x)[0])
show_img(py[0]>0, ax=ax, alpha=0.5);


# In[90]:


ax = show_img(denorm(x)[0])
show_img(y[0], ax=ax, alpha=0.5);


# ## 512x512

# In[19]:


TRAIN_DN = 'train'
MASKS_DN = 'train_masks_png'
sz = 512
bs = 16


# In[20]:


x_names = np.array([Path(TRAIN_DN)/o for o in masks_csv['img']])
y_names = np.array([Path(MASKS_DN)/f'{o[:-4]}_mask.png' for o in masks_csv['img']])


# In[21]:


((val_x,trn_x),(val_y,trn_y)) = split_by_idx(val_idxs, x_names, y_names)
len(val_x),len(trn_x)


# In[22]:


tfms = tfms_from_model(resnet34, sz, crop_type=CropType.NO, tfm_y=TfmType.CLASS, aug_tfms=aug_tfms)
datasets = ImageData.get_ds(MatchedFilesDataset, (trn_x,trn_y), (val_x,val_y), tfms, path=PATH)
md = ImageData(PATH, datasets, bs, num_workers=8, classes=None)


# In[60]:


denorm = md.trn_ds.denorm
x,y = next(iter(md.aug_dl))
x = denorm(x)


# In[61]:


fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for i,ax in enumerate(axes.flat):
    ax=show_img(x[i], ax=ax)
    show_img(y[i], ax=ax, alpha=0.5)
plt.tight_layout(pad=0.1)


# In[101]:


simple_up = nn.Sequential(
    nn.ReLU(),
    StdUpsample(512,256),
    StdUpsample(256,256),
    StdUpsample(256,256),
    StdUpsample(256,256),
    nn.ConvTranspose2d(256, 1, 2, stride=2),
    flatten_channel
)


# In[102]:


models = ConvnetBuilder(resnet34, 0, 0, 0, custom_head=simple_up)
learn = ConvLearner(md, models)
learn.opt_fn=optim.Adam
learn.crit=nn.BCEWithLogitsLoss()
learn.metrics=[accuracy_thresh(0.5)]


# In[103]:


learn.load('0')


# In[21]:


learn.lr_find()
learn.sched.plot()


# In[104]:


lr=4e-2


# In[105]:


learn.fit(lr,1,cycle_len=5,use_clr=(20,5))


# In[106]:


learn.save('tmp')


# In[120]:


learn.load('tmp')


# In[75]:


learn.unfreeze()
learn.bn_freeze(True)


# In[76]:


lrs = np.array([lr/100,lr/10,lr])/4


# In[77]:


learn.fit(lrs,1,cycle_len=8,use_clr=(20,8))


# In[78]:


learn.save('512')


# In[79]:


x,y = next(iter(md.val_dl))
py = to_np(learn.model(V(x)))


# In[80]:


ax = show_img(denorm(x)[0])
show_img(py[0]>0, ax=ax, alpha=0.5);


# In[81]:


ax = show_img(denorm(x)[0])
show_img(y[0], ax=ax, alpha=0.5);


# ## 1024x1024

# In[23]:


sz = 1024
bs = 4


# In[24]:


tfms = tfms_from_model(resnet34, sz, crop_type=CropType.NO, tfm_y=TfmType.CLASS, aug_tfms=aug_tfms)
datasets = ImageData.get_ds(MatchedFilesDataset, (trn_x,trn_y), (val_x,val_y), tfms, path=PATH)
md = ImageData(PATH, datasets, bs, num_workers=8, classes=None)


# In[25]:


denorm = md.trn_ds.denorm
x,y = next(iter(md.aug_dl))
x = denorm(x)
y = to_np(y)


# In[26]:


fig, axes = plt.subplots(2, 2, figsize=(8, 8))
for i,ax in enumerate(axes.flat):
    show_img(x[i], ax=ax)
    show_img(y[i], ax=ax, alpha=0.5)
plt.tight_layout(pad=0.1)


# In[27]:


simple_up = nn.Sequential(
    nn.ReLU(),
    StdUpsample(512,256),
    StdUpsample(256,256),
    StdUpsample(256,256),
    StdUpsample(256,256),
    nn.ConvTranspose2d(256, 1, 2, stride=2),
    flatten_channel,
)


# In[28]:


models = ConvnetBuilder(resnet34, 0, 0, 0, custom_head=simple_up)
learn = ConvLearner(md, models)
learn.opt_fn=optim.Adam
learn.crit=nn.BCEWithLogitsLoss()
learn.metrics=[accuracy_thresh(0.5)]


# In[29]:


learn.load('512')


# In[21]:


learn.lr_find()
learn.sched.plot()


# In[30]:


lr=4e-2


# In[31]:


learn.fit(lr,1,cycle_len=2,use_clr=(20,4))


# In[32]:


learn.save('tmp')


# In[97]:


learn.load('tmp')


# In[33]:


learn.unfreeze()
learn.bn_freeze(True)


# In[34]:


lrs = np.array([lr/100,lr/10,lr])/8


# In[102]:


learn.fit(lrs,1,cycle_len=40,use_clr=(20,10))


# In[103]:


learn.save('1024')


# In[104]:


x,y = next(iter(md.val_dl))
py = to_np(learn.model(V(x)))


# In[105]:


ax = show_img(denorm(x)[0])
show_img(py[0][0]>0, ax=ax, alpha=0.5);


# In[106]:


ax = show_img(denorm(x)[0])
show_img(y[0,...,-1], ax=ax, alpha=0.5);


# In[107]:


show_img(py[0][0]>0);


# In[108]:


show_img(y[0,...,-1]);


# ## Fin
