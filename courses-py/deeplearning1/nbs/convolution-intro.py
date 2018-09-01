
# coding: utf-8

# In[1]:


# get_ipython().magic(u'matplotlib inline')
import math,sys,os,numpy as np
from numpy.linalg import norm
from PIL import Image
from matplotlib import pyplot as plt, rcParams, rc
from scipy.ndimage import imread
from skimage.measure import block_reduce
import pickle
from scipy.ndimage.filters import correlate, convolve
from ipywidgets import interact, interactive, fixed
from ipywidgets.widgets import *
rc('animation', html='html5')
rcParams['figure.figsize'] = 3, 6
get_ipython().magic(u'precision 4')
np.set_printoptions(precision=4, linewidth=100)


# In[2]:


"""
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")
images, labels = mnist.train.images, mnist.train.labels
images = images.reshape((55000,28,28))
np.savez_compressed("/eee/MNIST_data/train", images=images, labels=labels)
"""


# In[3]:


"""
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")
images, labels = mnist.train.images, mnist.train.labels
images = images.reshape((55000,28,28))
np.savez_compressed("MNIST_data/train", images=images, labels=labels)
"""
1


# In[4]:


def plots(ims, interp=False, titles=None):
    ims=np.array(ims)
    mn,mx=ims.min(),ims.max()
    f = plt.figure(figsize=(12,24))
    for i in range(len(ims)):
        sp=f.add_subplot(1, len(ims), i+1)
        if not titles is None: sp.set_title(titles[i], fontsize=18)
        plt.imshow(ims[i], interpolation=None if interp else 'none', vmin=mn,vmax=mx)

def plot(im, interp=False):
    f = plt.figure(figsize=(3,6), frameon=True)
    plt.imshow(im, interpolation=None if interp else 'none')

plt.gray()
plt.close()


# In[5]:


data = np.load("/eee/MNIST_data/train.npz")
images=data['images']
labels=data['labels']
n=len(images)
images.shape


# In[6]:


plot(images[0])


# In[7]:


labels[0]


# In[8]:


plots(images[:5], titles=labels[:5])


# In[9]:


top=[[-1,-1,-1],
     [ 1, 1, 1],
     [ 0, 0, 0]]

plot(top)


# In[15]:


r=(0,28)
def zoomim(x1=0,x2=28,y1=0,y2=28):
    plot(images[0,y1:y2,x1:x2])
w=interactive(zoomim, x1=r,x2=r,y1=r,y2=r)
w


# In[16]:


k=w.kwargs
dims = np.index_exp[k['y1']:k['y2']:1,k['x1']:k['x2']]
images[0][dims]


# In[17]:


corrtop = correlate(images[0], top)


# In[18]:


corrtop[dims]


# In[19]:


plot(corrtop[dims])


# In[20]:


plot(corrtop)


# In[21]:


np.rot90(top, 1)


# In[22]:


convtop = convolve(images[0], np.rot90(top,2))
plot(convtop)
np.allclose(convtop, corrtop)


# In[23]:


straights=[np.rot90(top,i) for i in range(4)]
plots(straights)


# In[24]:


br=[[ 0, 0, 1],
    [ 0, 1,-1.5],
    [ 1,-1.5, 0]]

diags = [np.rot90(br,i) for i in range(4)]
plots(diags)


# In[25]:


rots = straights + diags
corrs = [correlate(images[0], rot) for rot in rots]
plots(corrs)


# In[ ]:


def pool(im): return block_reduce(im, (7,7), np.max)

plots([pool(im) for im in corrs])


# In[ ]:


eights=[images[i] for i in xrange(n) if labels[i]==8]
ones=[images[i] for i in xrange(n) if labels[i]==1]


# In[ ]:


plots(eights[:5])
plots(ones[:5])


# In[ ]:


pool8 = [np.array([pool(correlate(im, rot)) for im in eights]) for rot in rots]


# In[ ]:


len(pool8), pool8[0].shape


# In[ ]:


plots(pool8[0][0:5])


# In[ ]:


def normalize(arr): return (arr-arr.mean())/arr.std()


# In[ ]:


filts8 = np.array([ims.mean(axis=0) for ims in pool8])
filts8 = normalize(filts8)


# In[ ]:


plots(filts8)


# In[ ]:


pool1 = [np.array([pool(correlate(im, rot)) for im in ones]) for rot in rots]
filts1 = np.array([ims.mean(axis=0) for ims in pool1])
filts1 = normalize(filts1)


# In[ ]:


plots(filts1)


# In[ ]:


def pool_corr(im): return np.array([pool(correlate(im, rot)) for rot in rots])


# In[ ]:


plots(pool_corr(eights[0]))


# In[ ]:


def sse(a,b): return ((a-b)**2).sum()
def is8_n2(im): return 1 if sse(pool_corr(im),filts1) > sse(pool_corr(im),filts8) else 0


# In[ ]:


sse(pool_corr(eights[0]), filts8), sse(pool_corr(eights[0]), filts1)


# In[ ]:


[np.array([is8_n2(im) for im in ims]).sum() for ims in [eights,ones]]


# In[ ]:


[np.array([(1-is8_n2(im)) for im in ims]).sum() for ims in [eights,ones]]


# In[ ]:


def n1(a,b): return (np.fabs(a-b)).sum()
def is8_n1(im): return 1 if n1(pool_corr(im),filts1) > n1(pool_corr(im),filts8) else 0


# In[ ]:


[np.array([is8_n1(im) for im in ims]).sum() for ims in [eights,ones]]


# In[ ]:


[np.array([(1-is8_n1(im)) for im in ims]).sum() for ims in [eights,ones]]

