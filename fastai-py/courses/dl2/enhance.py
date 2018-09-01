
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ## Super resolution data

# In[2]:


from fastai.conv_learner import *
from pathlib import Path
torch.cuda.set_device(0)

torch.backends.cudnn.benchmark=True


# In[3]:


PATH = Path('data/imagenet')
PATH_TRN = PATH/'train'


# In[4]:


fnames_full,label_arr_full,all_labels = folder_source(PATH, 'train')
fnames_full = ['/'.join(Path(fn).parts[-2:]) for fn in fnames_full]
list(zip(fnames_full[:5],label_arr_full[:5]))


# In[5]:


all_labels[:5]


# In[6]:


np.random.seed(42)
keep_pct = 1.
# keep_pct = 0.02
keeps = np.random.rand(len(fnames_full)) < keep_pct
fnames = np.array(fnames_full, copy=False)[keeps]
label_arr = np.array(label_arr_full, copy=False)[keeps]


# In[7]:


arch = vgg16
sz_lr = 72


# In[8]:


scale,bs = 2,64
# scale,bs = 4,32
sz_hr = sz_lr*scale


# In[9]:


class MatchedFilesDataset(FilesDataset):
    def __init__(self, fnames, y, transform, path):
        self.y=y
        assert(len(fnames)==len(y))
        super().__init__(fnames, transform, path)
    def get_y(self, i): return open_image(os.path.join(self.path, self.y[i]))
    def get_c(self): return 0


# In[10]:


aug_tfms = [RandomDihedral(tfm_y=TfmType.PIXEL)]


# In[11]:


val_idxs = get_cv_idxs(len(fnames), val_pct=min(0.01/keep_pct, 0.1))
((val_x,trn_x),(val_y,trn_y)) = split_by_idx(val_idxs, np.array(fnames), np.array(fnames))
len(val_x),len(trn_x)


# In[12]:


img_fn = PATH/'train'/'n01558993'/'n01558993_9684.JPEG'


# In[13]:


tfms = tfms_from_model(arch, sz_lr, tfm_y=TfmType.PIXEL, aug_tfms=aug_tfms, sz_y=sz_hr)
datasets = ImageData.get_ds(MatchedFilesDataset, (trn_x,trn_y), (val_x,val_y), tfms, path=PATH_TRN)
md = ImageData(PATH, datasets, bs, num_workers=16, classes=None)


# In[14]:


denorm = md.val_ds.denorm


# In[15]:


def show_img(ims, idx, figsize=(5,5), normed=True, ax=None):
    if ax is None: fig,ax = plt.subplots(figsize=figsize)
    if normed: ims = denorm(ims)
    else:      ims = np.rollaxis(to_np(ims),1,4)
    ax.imshow(np.clip(ims,0,1)[idx])
    ax.axis('off')


# In[16]:


x,y = next(iter(md.val_dl))
x.size(),y.size()


# In[17]:


idx=1
fig,axes = plt.subplots(1, 2, figsize=(9,5))
show_img(x,idx, ax=axes[0])
show_img(y,idx, ax=axes[1])


# In[22]:


batches = [next(iter(md.aug_dl)) for i in range(9)]


# In[21]:


fig, axes = plt.subplots(3, 6, figsize=(18, 9))
for i,(x,y) in enumerate(batches):
    show_img(x,idx, ax=axes.flat[i*2])
    show_img(y,idx, ax=axes.flat[i*2+1])


# ## Model

# In[17]:


def conv(ni, nf, kernel_size=3, actn=False):
    layers = [nn.Conv2d(ni, nf, kernel_size, padding=kernel_size//2)]
    if actn: layers.append(nn.ReLU(True))
    return nn.Sequential(*layers)


# In[18]:


class ResSequential(nn.Module):
    def __init__(self, layers, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.m = nn.Sequential(*layers)

    def forward(self, x): return x + self.m(x) * self.res_scale


# In[19]:


def res_block(nf):
    return ResSequential(
        [conv(nf, nf, actn=True), conv(nf, nf)],
        0.1)


# In[20]:


def upsample(ni, nf, scale):
    layers = []
    for i in range(int(math.log(scale,2))):
        layers += [conv(ni, nf*4), nn.PixelShuffle(2)]
    return nn.Sequential(*layers)


# In[21]:


class SrResnet(nn.Module):
    def __init__(self, nf, scale):
        super().__init__()
        features = [conv(3, 64)]
        for i in range(8): features.append(res_block(64))
        features += [conv(64,64), upsample(64, 64, scale),
                     nn.BatchNorm2d(64),
                     conv(64, 3)]
        self.features = nn.Sequential(*features)
        
    def forward(self, x): return self.features(x)


# ## Pixel loss

# In[68]:


m = to_gpu(SrResnet(64, scale))
m = nn.DataParallel(m, [0,2])
learn = Learner(md, SingleModel(m), opt_fn=optim.Adam)
learn.crit = F.mse_loss


# In[69]:


learn.lr_find(start_lr=1e-5, end_lr=10000)
learn.sched.plot()


# In[70]:


lr=2e-3


# In[71]:


learn.fit(lr, 1, cycle_len=1, use_clr_beta=(40,10))


# In[72]:


x,y = next(iter(md.val_dl))
preds = learn.model(VV(x))


# In[77]:


idx=4
show_img(y,idx,normed=False)


# In[78]:


show_img(preds,idx,normed=False);


# In[80]:


show_img(x,idx,normed=True);


# In[133]:


x,y = next(iter(md.val_dl))
preds = learn.model(VV(x))


# In[134]:


show_img(y,idx,normed=False)


# In[129]:


show_img(preds,idx,normed=False);


# In[130]:


show_img(x,idx);


# ## Perceptual loss

# In[22]:


def icnr(x, scale=2, init=nn.init.kaiming_normal):
    new_shape = [int(x.shape[0] / (scale ** 2))] + list(x.shape[1:])
    subkernel = torch.zeros(new_shape)
    subkernel = init(subkernel)
    subkernel = subkernel.transpose(0, 1)
    subkernel = subkernel.contiguous().view(subkernel.shape[0],
                                            subkernel.shape[1], -1)
    kernel = subkernel.repeat(1, 1, scale ** 2)
    transposed_shape = [x.shape[1]] + [x.shape[0]] + list(x.shape[2:])
    kernel = kernel.contiguous().view(transposed_shape)
    kernel = kernel.transpose(0, 1)
    return kernel


# In[23]:


m_vgg = vgg16(True)

blocks = [i-1 for i,o in enumerate(children(m_vgg))
              if isinstance(o,nn.MaxPool2d)]
blocks, [m_vgg[i] for i in blocks]


# In[24]:


vgg_layers = children(m_vgg)[:23]
m_vgg = nn.Sequential(*vgg_layers).cuda().eval()
set_trainable(m_vgg, False)


# In[25]:


def flatten(x): return x.view(x.size(0), -1)


# In[26]:


class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = output
    def remove(self): self.hook.remove()        


# In[27]:


class FeatureLoss(nn.Module):
    def __init__(self, m, layer_ids, layer_wgts):
        super().__init__()
        self.m,self.wgts = m,layer_wgts
        self.sfs = [SaveFeatures(m[i]) for i in layer_ids]

    def forward(self, input, target, sum_layers=True):
        self.m(VV(target.data))
        res = [F.l1_loss(input,target)/100]
        targ_feat = [V(o.features.data.clone()) for o in self.sfs]
        self.m(input)
        res += [F.l1_loss(flatten(inp.features),flatten(targ))*wgt
               for inp,targ,wgt in zip(self.sfs, targ_feat, self.wgts)]
        if sum_layers: res = sum(res)
        return res
    
    def close(self):
        for o in self.sfs: o.remove()


# In[28]:


m = SrResnet(64, scale)


# In[29]:


conv_shuffle = m.features[10][0][0]
kernel = icnr(conv_shuffle.weight, scale=scale)
conv_shuffle.weight.data.copy_(kernel);


# In[30]:


m = to_gpu(m)


# In[30]:


learn = Learner(md, SingleModel(m), opt_fn=optim.Adam)


# In[31]:


t = torch.load(learn.get_model_path('sr-samp0'), map_location=lambda storage, loc: storage)
learn.model.load_state_dict(t, strict=False)


# In[32]:


learn.freeze_to(999)


# In[33]:


for i in range(10,13): set_trainable(m.features[i], True)


# In[34]:


conv_shuffle = m.features[10][2][0]
kernel = icnr(conv_shuffle.weight, scale=scale)
conv_shuffle.weight.data.copy_(kernel);


# In[31]:


m = nn.DataParallel(m, [0,2])
learn = Learner(md, SingleModel(m), opt_fn=optim.Adam)


# In[50]:


learn.set_data(md)


# In[32]:


learn.crit = FeatureLoss(m_vgg, blocks[:3], [0.2,0.7,0.1])


# In[33]:


lr=6e-3
wd=1e-7


# In[42]:


learn.lr_find(1e-4, 0.1, wds=wd, linear=True)


# In[43]:


learn.sched.plot(n_skip_end=1)


# In[34]:


learn.fit(lr, 1, cycle_len=2, wds=wd, use_clr=(20,10))


# In[35]:


learn.save('sr-samp0')


# In[37]:


learn.save('sr-samp1')


# In[91]:


learn.load('sr-samp1')


# In[41]:


lr=3e-3


# In[42]:


learn.fit(lr, 1, cycle_len=1, wds=wd, use_clr=(20,10))


# In[44]:


learn.save('sr-samp2')


# In[35]:


learn.unfreeze()


# In[40]:


learn.load('sr-samp2')


# In[41]:


learn.fit(lr/3, 1, cycle_len=1, wds=wd, use_clr=(20,10))


# In[42]:


learn.save('sr1')


# In[43]:


learn.sched.plot_loss()


# In[44]:


def plot_ds_img(idx, ax=None, figsize=(7,7), normed=True):
    if ax is None: fig,ax = plt.subplots(figsize=figsize)
    im = md.val_ds[idx][0]
    if normed: im = denorm(im)[0]
    else:      im = np.rollaxis(to_np(im),0,3)
    ax.imshow(im)
    ax.axis('off')


# In[45]:


fig,axes=plt.subplots(6,6,figsize=(20,20))
for i,ax in enumerate(axes.flat): plot_ds_img(i+200,ax=ax, normed=True)


# In[64]:


x,y=md.val_ds[215]


# In[65]:


y=y[None]


# In[66]:


learn.model.eval()
preds = learn.model(VV(x[None]))
x.shape,y.shape,preds.shape


# In[49]:


learn.crit(preds, V(y), sum_layers=False)


# In[50]:


learn.crit.close()


# In[67]:


_,axes=plt.subplots(1,2,figsize=(14,7))
show_img(x[None], 0, ax=axes[0])
show_img(preds,0, normed=True, ax=axes[1])

