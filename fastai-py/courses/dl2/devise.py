
# coding: utf-8

# In[72]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ## Devise

# In[73]:


from fastai.conv_learner import *
torch.backends.cudnn.benchmark=True

import fastText as ft


# In[5]:


import torchvision.transforms as transforms


# In[8]:


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

tfms = transforms.Compose([
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])


# In[9]:


fname = 'valid/n01440764/ILSVRC2012_val_00007197.JPEG'


# In[10]:


PATH = Path('data/imagenet/')
TMP_PATH = PATH/'tmp'
TRANS_PATH = Path('data/translate/')
PATH_TRN = PATH/'train'


# In[60]:


img = Image.open(PATH/fname)


# In[70]:


import fastai


# In[ ]:


fastai.dataloader.DataLoader


# In[71]:


arch=resnet50
ttfms,vtfms = tfms_from_model(arch, 224, transforms_side_on, max_zoom=1.1)
def to_array(x,y): return np.array(x).astype(np.float32)/255,None
def TT(x,y): return torch.from_numpy(x),None

ttfms.tfms = [to_array] + ttfms.tfms# + [TT]


# In[68]:


ttfms(img)


# In[89]:


ft_vecs = ft.load_model(str((TRANS_PATH/'wiki.en.bin')))


# In[106]:


ft_vecs.get_word_vector('king')


# In[91]:


np.corrcoef(ft_vecs.get_word_vector('jeremy'), ft_vecs.get_word_vector('Jeremy'))


# In[92]:


np.corrcoef(ft_vecs.get_word_vector('banana'), ft_vecs.get_word_vector('Jeremy'))


# ### Map imagenet classes to word vectors

# In[93]:


ft_words = ft_vecs.get_words(include_freq=True)
ft_word_dict = {k:v for k,v in zip(*ft_words)}
ft_words = sorted(ft_word_dict.keys(), key=lambda x: ft_word_dict[x])

len(ft_words)


# In[17]:


from fastai.io import get_data


# In[18]:


CLASSES_FN = 'imagenet_class_index.json'
get_data(f'http://files.fast.ai/models/{CLASSES_FN}', TMP_PATH/CLASSES_FN)


# In[19]:


WORDS_FN = 'classids.txt'
get_data(f'http://files.fast.ai/data/{WORDS_FN}', PATH/WORDS_FN)


# In[20]:


class_dict = json.load((TMP_PATH/CLASSES_FN).open())
classids_1k = dict(class_dict.values())
nclass = len(class_dict); nclass


# In[21]:


class_dict['0']


# In[22]:


classid_lines = (PATH/WORDS_FN).open().readlines()
classid_lines[:5]


# In[23]:


classids = dict(l.strip().split() for l in classid_lines)
len(classids),len(classids_1k)


# In[101]:


lc_vec_d = {w.lower(): ft_vecs.get_word_vector(w) for w in ft_words[-1000000:]}


# In[102]:


syn_wv = [(k, lc_vec_d[v.lower()]) for k,v in classids.items()
          if v.lower() in lc_vec_d]
syn_wv_1k = [(k, lc_vec_d[v.lower()]) for k,v in classids_1k.items()
          if v.lower() in lc_vec_d]
syn2wv = dict(syn_wv)
len(syn2wv)


# In[103]:


pickle.dump(syn2wv, (TMP_PATH/'syn2wv.pkl').open('wb'))
pickle.dump(syn_wv_1k, (TMP_PATH/'syn_wv_1k.pkl').open('wb'))


# In[11]:


syn2wv = pickle.load((TMP_PATH/'syn2wv.pkl').open('rb'))
syn_wv_1k = pickle.load((TMP_PATH/'syn_wv_1k.pkl').open('rb'))


# In[38]:


images = []
img_vecs = []

for d in (PATH/'train').iterdir():
    if d.name not in syn2wv: continue
    vec = syn2wv[d.name]
    for f in d.iterdir():
        images.append(str(f.relative_to(PATH)))
        img_vecs.append(vec)

n_val=0
for d in (PATH/'valid').iterdir():
    if d.name not in syn2wv: continue
    vec = syn2wv[d.name]
    for f in d.iterdir():
        images.append(str(f.relative_to(PATH)))
        img_vecs.append(vec)
        n_val += 1


# In[39]:


n_val


# In[13]:


img_vecs = np.stack(img_vecs)
img_vecs.shape


# In[41]:


pickle.dump(images, (TMP_PATH/'images.pkl').open('wb'))
pickle.dump(img_vecs, (TMP_PATH/'img_vecs.pkl').open('wb'))


# In[12]:


images = pickle.load((TMP_PATH/'images.pkl').open('rb'))
img_vecs = pickle.load((TMP_PATH/'img_vecs.pkl').open('rb'))


# In[24]:


arch = resnet50


# In[14]:


n = len(images); n


# In[15]:


val_idxs = list(range(n-28650, n))


# In[16]:


tfms = tfms_from_model(arch, 224, transforms_side_on, max_zoom=1.1)
md = ImageClassifierData.from_names_and_array(PATH, images, img_vecs, val_idxs=val_idxs,
        classes=None, tfms=tfms, continuous=True, bs=256)


# In[7]:


x,y = next(iter(md.val_dl))


# In[53]:


models = ConvnetBuilder(arch, md.c, is_multi=False, is_reg=True, xtra_fc=[1024], ps=[0.2,0.2])

learn = ConvLearner(md, models, precompute=True)
learn.opt_fn = partial(optim.Adam, betas=(0.9,0.99))


# In[54]:


def cos_loss(inp,targ): return 1 - F.cosine_similarity(inp,targ).mean()
learn.crit = cos_loss


# In[ ]:


learn.lr_find(start_lr=1e-4, end_lr=1e15)


# In[242]:


learn.sched.plot()


# In[55]:


lr = 1e-2
wd = 1e-7


# In[85]:


learn.precompute=True


# In[26]:


learn.fit(lr, 1, cycle_len=20, wds=wd, use_clr=(20,10))


# In[86]:


learn.bn_freeze(True)


# In[26]:


learn.fit(lr, 1, cycle_len=20, wds=wd, use_clr=(20,10))


# In[24]:


lrs = np.array([lr/1000,lr/100,lr])


# In[57]:


learn.precompute=False
learn.freeze_to(1)


# In[29]:


learn.save('pre0')


# In[58]:


learn.load('pre0')


# ## Image search

# ### Search imagenet classes

# In[88]:


syns, wvs = list(zip(*syn_wv_1k))
wvs = np.array(wvs)


# In[89]:


get_ipython().run_line_magic('time', 'pred_wv = learn.predict()')


# In[90]:


start=300


# In[91]:


denorm = md.val_ds.denorm

def show_img(im, figsize=None, ax=None):
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im)
    ax.axis('off')
    return ax

def show_imgs(ims, cols, figsize=None):
    fig,axes = plt.subplots(len(ims)//cols, cols, figsize=figsize)
    for i,ax in enumerate(axes.flat): show_img(ims[i], ax=ax)
    plt.tight_layout()


# In[92]:


show_imgs(denorm(md.val_ds[start:start+25][0]), 5, (10,10))


# In[94]:


import nmslib

def create_index(a):
    index = nmslib.init(space='angulardist')
    index.addDataPointBatch(a)
    index.createIndex()
    return index

def get_knns(index, vecs):
     return zip(*index.knnQueryBatch(vecs, k=10, num_threads=4))

def get_knn(index, vec): return index.knnQuery(vec, k=10)


# In[95]:


nn_wvs = create_index(wvs)


# In[96]:


idxs,dists = get_knns(nn_wvs, pred_wv)


# In[97]:


[[classids[syns[id]] for id in ids[:3]] for ids in idxs[start:start+10]]


# ### Search all wordnet noun classes

# In[98]:


all_syns, all_wvs = list(zip(*syn2wv.items()))
all_wvs = np.array(all_wvs)


# In[99]:


nn_allwvs = create_index(all_wvs)


# In[100]:


idxs,dists = get_knns(nn_allwvs, pred_wv)


# In[101]:


[[classids[all_syns[id]] for id in ids[:3]] for ids in idxs[start:start+10]]


# ### Text -> image search

# In[102]:


nn_predwv = create_index(pred_wv)


# In[103]:


en_vecd = pickle.load(open(TRANS_PATH/'wiki.en.pkl','rb'))


# In[104]:


vec = en_vecd['boat']


# In[105]:


idxs,dists = get_knn(nn_predwv, vec)
show_imgs([open_image(PATH/md.val_ds.fnames[i]) for i in idxs[:3]], 3, figsize=(9,3));


# In[106]:


vec = (en_vecd['engine'] + en_vecd['boat'])/2


# In[107]:


idxs,dists = get_knn(nn_predwv, vec)
show_imgs([open_image(PATH/md.val_ds.fnames[i]) for i in idxs[:3]], 3, figsize=(9,3));


# In[108]:


vec = (en_vecd['sail'] + en_vecd['boat'])/2


# In[79]:


idxs,dists = get_knn(nn_predwv, vec)
show_imgs([open_image(PATH/md.val_ds.fnames[i]) for i in idxs[:3]], 3, figsize=(9,3));


# ### Image->image

# In[80]:


fname = 'valid/n01440764/ILSVRC2012_val_00007197.JPEG'


# In[81]:


img = open_image(PATH/fname)


# In[82]:


show_img(img);


# In[83]:


t_img = md.val_ds.transform(img)
pred = learn.predict_array(t_img[None])


# In[84]:


idxs,dists = get_knn(nn_predwv, pred)
show_imgs([open_image(PATH/md.val_ds.fnames[i]) for i in idxs[1:4]], 3, figsize=(9,3));

