
# coding: utf-8

# # Generative Adversarial Networks in Keras

# In[1]:


get_ipython().magic(u'matplotlib inline')
import importlib
import utils2; importlib.reload(utils2)
from utils2 import *

from tqdm import tqdm


# ## The original GAN!

# See [this paper](https://arxiv.org/abs/1406.2661) for details of the approach we'll try first for our first GAN. We'll see if we can generate hand-drawn numbers based on MNIST, so let's load that dataset first.
# 
# We'll be refering to the discriminator as 'D' and the generator as 'G'.

# In[2]:


from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape


# In[3]:


n = len(X_train)


# In[4]:


X_train = X_train.reshape(n, -1).astype(np.float32)
X_test = X_test.reshape(len(X_test), -1).astype(np.float32)


# In[5]:


X_train /= 255.; X_test /= 255.


# ## Train

# This is just a helper to plot a bunch of generated images.

# In[6]:


def plot_gen(G, n_ex=16):
    plot_multi(G.predict(noise(n_ex)).reshape(n_ex, 28,28), cmap='gray')


# Create some random data for the generator.

# In[191]:


def noise(bs): return np.random.rand(bs,100)


# Create a batch of some real and some generated data, with appropriate labels, for the discriminator.

# In[135]:


def data_D(sz, G):
    real_img = X_train[np.random.randint(0,n,size=sz)]
    X = np.concatenate((real_img, G.predict(noise(sz))))
    return X, [0]*sz + [1]*sz


# In[136]:


def make_trainable(net, val):
    net.trainable = val
    for l in net.layers: l.trainable = val


# Train a few epochs, and return the losses for D and G. In each epoch we:
# 
# 1. Train D on one batch from data_D()
# 2. Train G to create images that the discriminator predicts as real.

# In[192]:


def train(D, G, m, nb_epoch=5000, bs=128):
    dl,gl=[],[]
    for e in tqdm(range(nb_epoch)):
        X,y = data_D(bs//2, G)
        dl.append(D.train_on_batch(X,y))
        make_trainable(D, False)
        gl.append(m.train_on_batch(noise(bs), np.zeros([bs])))
        make_trainable(D, True)
    return dl,gl


# ## MLP GAN

# We'll keep thinks simple by making D & G plain ole' MLPs.

# In[166]:


MLP_G = Sequential([
    Dense(200, input_shape=(100,), activation='relu'),
    Dense(400, activation='relu'),
    Dense(784, activation='sigmoid'),
])


# In[168]:


MLP_D = Sequential([
    Dense(300, input_shape=(784,), activation='relu'),
    Dense(300, activation='relu'),
    Dense(1, activation='sigmoid'),
])
MLP_D.compile(Adam(1e-4), "binary_crossentropy")


# In[169]:


MLP_m = Sequential([MLP_G,MLP_D])
MLP_m.compile(Adam(1e-4), "binary_crossentropy")


# In[160]:


dl,gl = train(MLP_D, MLP_G, MLP_m, 8000)


# The loss plots for most GANs are nearly impossible to interpret - which is one of the things that make them hard to train.

# In[161]:


plt.plot(dl[100:])


# In[162]:


plt.plot(gl[100:])


# This is what's known in the literature as "mode collapse".

# In[165]:


plot_gen()


# OK, so that didn't work. Can we do better?...

# ## DCGAN

# There's lots of ideas out there to make GANs train better, since they are notoriously painful to get working. The [paper introducing DCGANs](https://arxiv.org/abs/1511.06434) is the main basis for our next section. Add see https://github.com/soumith/ganhacks for many tips!
# 
# Because we're using a CNN from now on, we'll reshape our digits into proper images.

# In[41]:


X_train = X_train.reshape(n, 28, 28, 1)
X_test = X_test.reshape(len(X_test), 28, 28, 1)


# Our generator uses a number of upsampling steps as suggested in the above papers. We use nearest neighbor upsampling rather than fractionally strided convolutions, as discussed in our style transfer notebook.

# In[250]:


CNN_G = Sequential([
    Dense(512*7*7, input_dim=100, activation=LeakyReLU()),
    BatchNormalization(mode=2),
    Reshape((7, 7, 512)),
    UpSampling2D(),
    Convolution2D(64, 3, 3, border_mode='same', activation=LeakyReLU()),
    BatchNormalization(mode=2),
    UpSampling2D(),
    Convolution2D(32, 3, 3, border_mode='same', activation=LeakyReLU()),
    BatchNormalization(mode=2),
    Convolution2D(1, 1, 1, border_mode='same', activation='sigmoid')
])


# The discriminator uses a few downsampling steps through strided convolutions.

# In[251]:


CNN_D = Sequential([
    Convolution2D(256, 5, 5, subsample=(2,2), border_mode='same', 
                  input_shape=(28, 28, 1), activation=LeakyReLU()),
    Convolution2D(512, 5, 5, subsample=(2,2), border_mode='same', activation=LeakyReLU()),
    Flatten(),
    Dense(256, activation=LeakyReLU()),
    Dense(1, activation = 'sigmoid')
])

CNN_D.compile(Adam(1e-3), "binary_crossentropy")


# We train D a "little bit" so it can at least tell a real image from random noise.

# In[252]:


sz = n//200
x1 = np.concatenate([np.random.permutation(X_train)[:sz], CNN_G.predict(noise(sz))])
CNN_D.fit(x1, [0]*sz + [1]*sz, batch_size=128, nb_epoch=1, verbose=2)


# In[253]:


CNN_m = Sequential([CNN_G, CNN_D])
CNN_m.compile(Adam(1e-4), "binary_crossentropy")


# In[261]:


K.set_value(CNN_D.optimizer.lr, 1e-3)
K.set_value(CNN_m.optimizer.lr, 1e-3)


# Now we can train D & G iteratively.

# In[262]:


dl,gl = train(CNN_D, CNN_G, CNN_m, 2500)


# In[259]:


plt.plot(dl[10:])


# In[260]:


plt.plot(gl[10:])


# Better than our first effort, but still a lot to be desired:...

# In[258]:


plot_gen(CNN_G)


# ## End
