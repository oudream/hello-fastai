
# coding: utf-8

# # Dogs breeds
# 
# https://youtu.be/JNxcznsrRb8?t=1h31m8s

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.imports import *
from fastai.torch_imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *


# In[ ]:


#torch.cuda.set_device(0)


# Kaggle Dog Breed Identification. Get data from https://www.kaggle.com/c/dog-breed-identification

# In[ ]:


PATH = "/eee/dogbreed/"
sz = 224
arch = resnext101_64
bs = 58


# In[ ]:


label_csv = f'{PATH}labels.csv'
n = len(list(open(label_csv))) - 1 # header is not counted (-1)
val_idxs = get_cv_idxs(n) # random 20% data for validation set


# In[ ]:


n


# In[ ]:


len(val_idxs)


# In[ ]:


# If you haven't downloaded weights.tgz yet, download the file.
#     http://forums.fast.ai/t/error-when-trying-to-use-resnext50/7555
#     http://forums.fast.ai/t/lesson-2-in-class-discussion/7452/222
#!wget -O fastai/weights.tgz http://files.fast.ai/models/weights.tgz

#!tar xvfz fastai/weights.tgz -C fastai


# ## Initial exploration

# In[ ]:


get_ipython().system('ls {PATH}')


# In[ ]:


label_df = pd.read_csv(label_csv)


# In[ ]:


label_df.head()


# In[ ]:


label_df.pivot_table(index="breed", aggfunc=len).sort_values('id', ascending=False)


# In[ ]:


tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
data = ImageClassifierData.from_csv(PATH, 'train', f'{PATH}labels.csv', test_name='test', # we need to specify where the test set is if you want to submit to Kaggle competitions
                                   val_idxs=val_idxs, suffix='.jpg', tfms=tfms, bs=bs)


# In[ ]:


fn = PATH + data.trn_ds.fnames[0]; fn


# In[ ]:


img = PIL.Image.open(fn); img


# In[ ]:


img.size


# In[ ]:


size_d = {k: PIL.Image.open(PATH + k).size for k in data.trn_ds.fnames}


# In[ ]:


row_sz, col_sz = list(zip(*size_d.values()))


# In[ ]:


row_sz = np.array(row_sz); col_sz = np.array(col_sz)


# In[ ]:


row_sz[:5]


# In[ ]:


plt.hist(row_sz);


# In[ ]:


plt.hist(row_sz[row_sz < 1000])


# In[ ]:


plt.hist(col_sz);


# In[ ]:


plt.hist(col_sz[col_sz < 1000])


# In[ ]:


len(data.trn_ds), len(data.test_ds)


# In[ ]:


len(data.classes), data.classes[:5]


# ## Initial model

# In[ ]:


def get_data(sz, bs): # sz: image size, bs: batch size
    tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
    data = ImageClassifierData.from_csv(PATH, 'train', f'{PATH}labels.csv', test_name='test',
                                       val_idxs=val_idxs, suffix='.jpg', tfms=tfms, bs=bs)
    
    # http://forums.fast.ai/t/how-to-train-on-the-full-dataset-using-imageclassifierdata-from-csv/7761/13
    # http://forums.fast.ai/t/how-to-train-on-the-full-dataset-using-imageclassifierdata-from-csv/7761/37
    return data if sz > 300 else data.resize(340, 'tmp') # Reading the jpgs and resizing is slow for big images, so resizing them all to 340 first saves time

#Source:   
#    def resize(self, targ, new_path):
#        new_ds = []
#        dls = [self.trn_dl,self.val_dl,self.fix_dl,self.aug_dl]
#        if self.test_dl: dls += [self.test_dl, self.test_aug_dl]
#        else: dls += [None,None]
#        t = tqdm_notebook(dls)
#        for dl in t: new_ds.append(self.resized(dl, targ, new_path))
#        t.close()
#        return self.__class__(new_ds[0].path, new_ds, self.bs, self.num_workers, self.classes)
#File:      ~/fastai/courses/dl1/fastai/dataset.py


# ### Precompute

# In[ ]:


data = get_data(sz, bs)


# In[ ]:


learn = ConvLearner.pretrained(arch, data, precompute=True)


# In[ ]:


learn.fit(1e-2, 5)


# ### Augment

# In[ ]:


from sklearn import metrics


# In[ ]:


data = get_data(sz, bs)


# In[ ]:


learn = ConvLearner.pretrained(arch, data, precompute=True, ps=0.5)


# In[ ]:


learn.fit(1e-2, 2)


# In[ ]:


learn.precompute = False


# In[ ]:


learn.fit(1e-2, 5, cycle_len=1)


# In[ ]:


learn.save('224_pre')


# In[ ]:


learn.load('224_pre')


# ## Increase size

# In[ ]:


# Starting training on small images for a few epochs, then switching to bigger images, and continuing training is an amazingly effective way to avoid overfitting.

# http://forums.fast.ai/t/planet-classification-challenge/7824/96
# set_data doesn’t change the model at all. It just gives it new data to train with.
learn.set_data(get_data(299, bs)) 
learn.freeze()

#Source:   
#    def set_data(self, data, precompute=False):
#        super().set_data(data)
#        if precompute:
#            self.unfreeze()
#            self.save_fc1()
#            self.freeze()
#            self.precompute = True
#        else:
#            self.freeze()
#File:      ~/fastai/courses/dl1/fastai/conv_learner.py


# In[ ]:


learn.summary()


# In[ ]:


learn.fit(1e-2, 3, cycle_len=1)


# Validation loss is much lower than training loss. This is a sign of underfitting. Cycle_len=1 may be too short. Let's set cycle_mult=2 to find better parameter.

# In[ ]:


# When you are under fitting, it means cycle_len=1 is too short (learning rate is getting reset before it had the chance to zoom in properly).
learn.fit(1e-2, 3, cycle_len=1, cycle_mult=2) # 1+2+4 = 7 epochs


# Training loss and validation loss are getting closer and smaller. We are on right track.

# In[ ]:


log_preds, y = learn.TTA() # (5, 2044, 120), (2044,)
probs = np.mean(np.exp(log_preds),0)
accuracy_np(probs, y), metrics.log_loss(y, probs)


# In[ ]:


len(data.val_ds.y), data.val_ds.y[:5]


# In[ ]:


learn.save('299_pre')


# In[ ]:


learn.load('299_pre')


# In[ ]:


learn.fit(1e-2, 1, cycle_len=2) # 1+1 = 2 epochs


# In[ ]:


learn.save('299_pre')


# In[ ]:


log_preds, y = learn.TTA()
probs = np.mean(np.exp(log_preds),0)
accuracy_np(probs, y), metrics.log_loss(y, probs)


# This dataset is so similar to ImageNet dataset. Training convolution layers doesn't help much. We are not going to unfreeze.

# ## Create submission
# 
# https://youtu.be/9C06ZPF8Uuc?t=1905

# In[ ]:


data.classes


# In[ ]:


data.test_ds.fnames


# In[ ]:


log_preds, y = learn.TTA(is_test=True) # use test dataset rather than validation dataset
probs = np.mean(np.exp(log_preds),0)
#accuracy_np(probs, y), metrcs.log_loss(y, probs) # This does not make sense since test dataset has no labels


# In[ ]:


probs.shape # (n_images, n_classes)


# In[ ]:


df = pd.DataFrame(probs)
df.columns = data.classes


# In[ ]:


df.insert(0, 'id', [o[5:-4] for o in data.test_ds.fnames])


# In[ ]:


df.head()


# In[ ]:


SUBM = f'{PATH}/subm/'
os.makedirs(SUBM, exist_ok=True)
df.to_csv(f'{SUBM}subm.gz', compression='gzip', index=False)


# In[ ]:


FileLink(f'{SUBM}subm.gz')


# ## Individual prediction

# In[ ]:


fn = data.val_ds.fnames[0]
fn


# In[ ]:


Image.open(PATH + fn).resize((150, 150))


# In[ ]:


# Method 1.
trn_tfms, val_tfms = tfms_from_model(arch, sz)
ds = FilesIndexArrayDataset([fn], np.array([0]), val_tfms, PATH)
dl = DataLoader(ds)
preds = learn.predict_dl(dl)
np.argmax(preds)


# In[ ]:


learn.data.classes[np.argmax(preds)]


# In[ ]:


# Method 2.
trn_tfms, val_tfms = tfms_from_model(arch, sz)
im = val_tfms(open_image(PATH + fn)) # open_image() returns numpy.ndarray
preds = learn.predict_array(im[None])
np.argmax(preds)

