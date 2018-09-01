
# coding: utf-8

# # Linear models with CNN features

# In[15]:


# Rather than importing everything manually, we'll make things easy
#   and load them all in utils.py, and just import them from there.
get_ipython().magic(u'matplotlib inline')
import utils; reload(utils)
from utils import *


# ## Introduction

# We need to find a way to convert the imagenet predictions to a probability of being a cat or a dog, since that is what the Kaggle competition requires us to submit. We could use the imagenet hierarchy to download a list of all the imagenet categories in each of the dog and cat groups, and could then solve our problem in various ways, such as:
# 
# - Finding the largest probability that's either a cat or a dog, and using that label
# - Averaging the probability of all the cat categories and comparing it to the average of all the dog categories.
# 
# But these approaches have some downsides:
# 
# - They require manual coding for something that we should be able to learn from the data
# - They ignore information available in the predictions; for instance, if the models predicts that there is a bone in the image, it's more likely to be a dog than a cat.
# 
# A very simple solution to both of these problems is to learn a linear model that is trained using the 1,000 predictions from the imagenet model for each image as input, and the dog/cat label as target.

# In[2]:


get_ipython().magic(u'matplotlib inline')
from __future__ import division,print_function
import os, json
from glob import glob
import numpy as np
import scipy
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
np.set_printoptions(precision=4, linewidth=100)
from matplotlib import pyplot as plt
import utils; reload(utils)
from utils import plots, get_batches, plot_confusion_matrix, get_data


# In[3]:


from numpy.random import random, permutation
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom

import keras
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.models import Sequential
from keras.layers import Input
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop
from keras.preprocessing import image


# ## Linear models in keras

# It turns out that each of the Dense() layers is just a *linear model*, followed by a simple *activation function*. We'll learn about the activation function later - first, let's review how linear models work.
# 
# A linear model is (as I'm sure you know) simply a model where each row is calculated as *sum(row &#42; weights)*, where *weights* needs to be learnt from the data, and will be the same for every row. For example, let's create some data that we know is linearly related:

# In[3]:


x = random((30,2))
y = np.dot(x, [2., 3.]) + 1.


# In[4]:


x[:5]


# In[5]:


y[:5]


# We can use keras to create a simple linear model (*Dense()* - with no activation - in Keras) and optimize it using SGD to minimize mean squared error (*mse*):

# In[6]:


lm = Sequential([ Dense(1, input_shape=(2,)) ])
lm.compile(optimizer=SGD(lr=0.1), loss='mse')


# (See the *Optim Tutorial* notebook and associated Excel spreadsheet to learn all about SGD and related optimization algorithms.)
# 
# This has now learnt internal weights inside the lm model, which we can use to evaluate the loss function (MSE).

# In[8]:


lm.evaluate(x, y, verbose=0)


# In[10]:


lm.fit(x, y, nb_epoch=5, batch_size=1)


# In[11]:


lm.evaluate(x, y, verbose=0)


# And, of course, we can also take a look at the weights - after fitting, we should see that they are close to the weights we used to calculate y (2.0, 3.0, and 1.0).

# In[12]:


lm.get_weights()


# ## Train linear model on predictions

# Using a Dense() layer in this way, we can easily convert the 1,000 predictions given by our model into a probability of dog vs cat--simply train a linear model to take the 1,000 predictions as input, and return dog or cat as output, learning from the Kaggle data. This should be easier and more accurate than manually creating a map from imagenet categories to one dog/cat category. 

# ### Training the model

# We start with some basic config steps. We copy a small amount of our data into a 'sample' directory, with the exact same structure as our 'train' directory--this is *always* a good idea in *all* machine learning, since we should do all of our initial testing using a dataset small enough that we never have to wait for it.

# In[16]:


path = "data/dogscats/sample/"
# path = "data/dogscats/"
model_path = path + 'models/'
if not os.path.exists(model_path): os.mkdir(model_path)


# We will process as many images at a time as our graphics card allows. This is a case of trial and error to find the max batch size - the largest size that doesn't give an out of memory error.

# In[17]:


# batch_size=100
batch_size=4


# We need to start with our VGG 16 model, since we'll be using its predictions and features.

# In[18]:


from vgg16 import Vgg16
vgg = Vgg16()
model = vgg.model


# Our overall approach here will be:
# 
# 1. Get the true labels for every image
# 2. Get the 1,000 imagenet category predictions for every image
# 3. Feed these predictions as input to a simple linear model.
# 
# Let's start by grabbing training and validation batches.

# In[22]:


# Use batch size of 1 since we're just doing preprocessing on the CPU
val_batches = get_batches(path+'valid', shuffle=False, batch_size=1)
batches = get_batches(path+'train', shuffle=False, batch_size=1)


# Loading and resizing the images every time we want to use them isn't necessary - instead we should save the processed arrays. By far the fastest way to save and load numpy arrays is using bcolz. This also compresses the arrays, so we save disk space. Here are the functions we'll use to save and load using bcolz.

# In[8]:


import bcolz
def save_array(fname, arr): c=bcolz.carray(arr, rootdir=fname, mode='w'); c.flush()
def load_array(fname): return bcolz.open(fname)[:]


# We have provided a simple function that joins the arrays from all the batches - let's use this to grab the training and validation data:

# In[ ]:


val_data = get_data(path+'valid')


# In[231]:


trn_data = get_data(path+'train')


# In[155]:


trn_data.shape


# In[153]:


save_array(model_path+'train_data.bc', trn_data)
save_array(model_path+'valid_data.bc', val_data)


# We can load our training and validation data later without recalculating them:

# In[19]:


trn_data = load_array(model_path+'train_data.bc')
val_data = load_array(model_path+'valid_data.bc')


# In[23]:


val_data.shape


# Keras returns *classes* as a single column, so we convert to one hot encoding

# In[20]:


def onehot(x): return np.array(OneHotEncoder().fit_transform(x.reshape(-1,1)).todense())


# In[23]:


val_classes = val_batches.classes
trn_classes = batches.classes
val_labels = onehot(val_classes)
trn_labels = onehot(trn_classes)


# In[27]:


trn_labels.shape


# In[24]:


trn_classes[:4]


# In[28]:


trn_labels[:4]


# ...and their 1,000 imagenet probabilties from VGG16--these will be the *features* for our linear model:

# In[144]:


trn_features = model.predict(trn_data, batch_size=batch_size)
val_features = model.predict(val_data, batch_size=batch_size)


# In[26]:


trn_features.shape


# In[149]:


save_array(model_path+'train_lastlayer_features.bc', trn_features)
save_array(model_path+'valid_lastlayer_features.bc', val_features)


# We can load our training and validation features later without recalculating them:

# In[25]:


trn_features = load_array(model_path+'train_lastlayer_features.bc')
val_features = load_array(model_path+'valid_lastlayer_features.bc')


# Now we can define our linear model, just like we did earlier:

# In[28]:


# 1000 inputs, since that's the saved features, and 2 outputs, for dog and cat
lm = Sequential([ Dense(2, activation='softmax', input_shape=(1000,)) ])
lm.compile(optimizer=RMSprop(lr=0.1), loss='categorical_crossentropy', metrics=['accuracy'])


# We're ready to fit the model!

# In[29]:


batch_size=64


# In[12]:


batch_size=4


# In[32]:


lm.fit(trn_features, trn_labels, nb_epoch=3, batch_size=batch_size, 
       validation_data=(val_features, val_labels))


# In[31]:


lm.summary()


# ### Viewing model prediction examples

# Keras' *fit()* function conveniently shows us the value of the loss function, and the accuracy, after every epoch ("*epoch*" refers to one full run through all training examples). The most important metrics for us to look at are for the validation set, since we want to check for over-fitting. 
# 
# - **Tip**: with our first model we should try to overfit before we start worrying about how to handle that - there's no point even thinking about regularization, data augmentation, etc if you're still under-fitting! (We'll be looking at these techniques shortly).
# 
# As well as looking at the overall metrics, it's also a good idea to look at examples of each of:
# 1. A few correct labels at random
# 2. A few incorrect labels at random
# 3. The most correct labels of each class (ie those with highest probability that are correct)
# 4. The most incorrect labels of each class (ie those with highest probability that are incorrect)
# 5. The most uncertain labels (ie those with probability closest to 0.5).
# 
# Let's see what we, if anything, we can from these (in general, these are particularly useful for debugging problems in the model; since this model is so simple, there may not be too much to learn at this stage.)

# Calculate predictions on validation set, so we can find correct and incorrect examples:

# In[37]:


# We want both the classes...
preds = lm.predict_classes(val_features, batch_size=batch_size)
# ...and the probabilities of being a cat
probs = lm.predict_proba(val_features, batch_size=batch_size)[:,0]
probs[:8]


# In[38]:


preds[:8]


# Get the filenames for the validation set, so we can view images:

# In[39]:


filenames = val_batches.filenames


# In[40]:


# Number of images to view for each visualization task
n_view = 4


# Helper function to plot images by index in the validation set:

# In[41]:


def plots_idx(idx, titles=None):
    plots([image.load_img(path + 'valid/' + filenames[i]) for i in idx], titles=titles)


# In[42]:


#1. A few correct labels at random
correct = np.where(preds==val_labels[:,1])[0]
idx = permutation(correct)[:n_view]
plots_idx(idx, probs[idx])


# In[43]:


#2. A few incorrect labels at random
incorrect = np.where(preds!=val_labels[:,1])[0]
idx = permutation(incorrect)[:n_view]
plots_idx(idx, probs[idx])


# In[44]:


#3. The images we most confident were cats, and are actually cats
correct_cats = np.where((preds==0) & (preds==val_labels[:,1]))[0]
most_correct_cats = np.argsort(probs[correct_cats])[::-1][:n_view]
plots_idx(correct_cats[most_correct_cats], probs[correct_cats][most_correct_cats])


# In[45]:


# as above, but dogs
correct_dogs = np.where((preds==1) & (preds==val_labels[:,1]))[0]
most_correct_dogs = np.argsort(probs[correct_dogs])[:n_view]
plots_idx(correct_dogs[most_correct_dogs], 1-probs[correct_dogs][most_correct_dogs])


# In[46]:


#3. The images we were most confident were cats, but are actually dogs
incorrect_cats = np.where((preds==0) & (preds!=val_labels[:,1]))[0]
most_incorrect_cats = np.argsort(probs[incorrect_cats])[::-1][:n_view]
if len(most_incorrect_cats):
    plots_idx(incorrect_cats[most_incorrect_cats], probs[incorrect_cats][most_incorrect_cats])
else:
    print('No incorrect cats!')


# In[47]:


#3. The images we were most confident were dogs, but are actually cats
incorrect_dogs = np.where((preds==1) & (preds!=val_labels[:,1]))[0]
most_incorrect_dogs = np.argsort(probs[incorrect_dogs])[:n_view]
if len(most_incorrect_dogs):
    plots_idx(incorrect_dogs[most_incorrect_dogs], 1-probs[incorrect_dogs][most_incorrect_dogs])
else:
    print('No incorrect dogs!')


# In[48]:


#5. The most uncertain labels (ie those with probability closest to 0.5).
most_uncertain = np.argsort(np.abs(probs-0.5))
plots_idx(most_uncertain[:n_view], probs[most_uncertain])


# Perhaps the most common way to analyze the result of a classification model is to use a [confusion matrix](http://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/). Scikit-learn has a convenient function we can use for this purpose:

# In[49]:


cm = confusion_matrix(val_classes, preds)


# We can just print out the confusion matrix, or we can show a graphical view (which is mainly useful for dependents with a larger number of categories).

# In[50]:


plot_confusion_matrix(cm, val_batches.class_indices)


# ### About activation functions

# Do you remember how we defined our linear model? Here it is again for reference:
# 
# ```python
# lm = Sequential([ Dense(2, activation='softmax', input_shape=(1000,)) ])
# ```
# 
# And do you remember the definition of a fully connected layer in the original VGG?:
# 
# ```python
# model.add(Dense(4096, activation='relu'))
# ```
# 
# You might we wondering, what's going on with that *activation* parameter? Adding an 'activation' parameter to a layer in Keras causes an additional function to be called after the layer is calculated. You'll recall that we had no such parameter in our most basic linear model at the start of this lesson - that's because a simple linear model has no *activation function*. But nearly all deep model layers have an activation function - specifically, a *non-linear* activation function, such as tanh, sigmoid (```1/(1+exp(x))```), or relu (```max(0,x)```, called the *rectified linear* function). Why?
# 
# The reason for this is that if you stack purely linear layers on top of each other, then you just end up with a linear layer! For instance, if your first layer was ```2*x```, and your second was ```-2*x```, then the combination is: ```-2*(2*x) = -4*x```. If that's all we were able to do with deep learning, it wouldn't be very deep! But what if we added a relu activation after our first layer? Then the combination would be: ```-2 * max(0, 2*x)```. As you can see, that does not simplify to just a linear function like the previous example--and indeed we can stack as many of these on top of each other as we wish, to create arbitrarily complex functions.
# 
# And why would we want to do that? Because it turns out that such a stack of linear functions and non-linear activations can approximate any other function just as close as we want. So we can **use it to model anything**! This extraordinary insight is known as the *universal approximation theorem*. For a visual understanding of how and why this works, I strongly recommend you read Michael Nielsen's [excellent interactive visual tutorial](http://neuralnetworksanddeeplearning.com/chap4.html).

# The last layer generally needs a different activation function to the other layers--because we want to encourage the last layer's output to be of an appropriate form for our particular problem. For instance, if our output is a one hot encoded categorical variable, we want our final layer's activations to add to one (so they can be treated as probabilities) and to have generally a single activation much higher than the rest (since with one hot encoding we have just a single 'one', and all other target outputs are zero). Our classication problems will always have this form, so we will introduce the activation function that has these properties: the *softmax* function. Softmax is defined as (for the i'th output activation): ```exp(x[i]) / sum(exp(x))```.
# 
# I suggest you try playing with that function in a spreadsheet to get a sense of how it behaves.
# 
# We will see other activation functions later in this course - but relu (and minor variations) for intermediate layers and softmax for output layers will be by far the most common.

# # Modifying the model

# ## Retrain last layer's linear model

# Since the original VGG16 network's last layer is Dense (i.e. a linear model) it seems a little odd that we are adding an additional linear model on top of it. This is especially true since the last layer had a softmax activation, which is an odd choice for an intermediate layer--and by adding an extra layer on top of it, we have made it an intermediate layer. What if we just removed the original final layer and replaced it with one that we train for the purpose of distinguishing cats and dogs? It turns out that this is a good idea - as we'll see!
# 
# We start by removing the last layer, and telling Keras that we want to fix the weights in all the other layers (since we aren't looking to learn new parameters for those other layers).

# In[34]:


vgg.model.summary()


# In[15]:


model.pop()
for layer in model.layers: layer.trainable=False


# **Careful!** Now that we've modified the definition of *model*, be careful not to rerun any code in the previous sections, without first recreating the model from scratch! (Yes, I made that mistake myself, which is why I'm warning you about it now...)
# 
# Now we're ready to add our new final layer...

# In[16]:


model.add(Dense(2, activation='softmax'))


# In[35]:


get_ipython().magic(u'pinfo2 vgg.finetune')


# ...and compile our updated model, and set up our batches to use the preprocessed images (note that now we will also *shuffle* the training batches, to add more randomness when using multiple epochs):

# In[17]:


gen=image.ImageDataGenerator()
batches = gen.flow(trn_data, trn_labels, batch_size=batch_size, shuffle=True)
val_batches = gen.flow(val_data, val_labels, batch_size=batch_size, shuffle=False)


# We'll define a simple function for fitting models, just to save a little typing...

# In[18]:


def fit_model(model, batches, val_batches, nb_epoch=1):
    model.fit_generator(batches, samples_per_epoch=batches.n, nb_epoch=nb_epoch, 
                        validation_data=val_batches, nb_val_samples=val_batches.n)


# ...and now we can use it to train the last layer of our model!
# 
# (It runs quite slowly, since it still has to calculate all the previous layers in order to know what input to pass to the new final layer. We could precalculate the output of the penultimate layer, like we did for the final layer earlier - but since we're only likely to want one or two iterations, it's easier to follow this alternative approach.)

# In[19]:


opt = RMSprop(lr=0.1)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


# In[166]:


fit_model(model, batches, val_batches, nb_epoch=2)


# Before moving on, go back and look at how little code we had to write in this section to finetune the model. Because this is such an important and common operation, keras is set up to make it as easy as possible. We didn't even have to use any external helper functions in this section.

# It's a good idea to save weights of all your models, so you can re-use them later. Be sure to note the git log number of your model when keeping a research journal of your results.

# In[167]:


model.save_weights(model_path+'finetune1.h5')


# In[20]:


model.load_weights(model_path+'finetune1.h5')


# In[23]:


model.evaluate(val_data, val_labels)


# We can look at the earlier prediction examples visualizations by redefining *probs* and *preds* and re-using our earlier code.

# In[168]:


preds = model.predict_classes(val_data, batch_size=batch_size)
probs = model.predict_proba(val_data, batch_size=batch_size)[:,0]
probs[:8]


# In[178]:


cm = confusion_matrix(val_classes, preds)


# In[180]:


plot_confusion_matrix(cm, {'cat':0, 'dog':1})


# ## Retraining more layers

# Now that we've fine-tuned the new final layer, can we, and should we, fine-tune *all* the dense layers? The answer to both questions, it turns out, is: yes! Let's start with the "can we" question...

# ### An introduction to back-propagation

# The key to training multiple layers of a model, rather than just one, lies in a technique called "back-propagation" (or *backprop* to its friends). Backprop is one of the many words in deep learning parlance that is creating a new word for something that already exists - in this case, backprop simply refers to calculating gradients using the *chain rule*. (But we will still introduce the deep learning terms during this course, since it's important to know them when reading about or discussing deep learning.)
# 
# As you (hopefully!) remember from high school, the chain rule is how you calculate the gradient of a "function of a function"--something of the form *f(u), where u=g(x)*. For instance, let's say your function is ```pow((2*x), 2)```. Then u is ```2*x```, and f(u) is ```power(u, 2)```. The chain rule tells us that the derivative of this is simply the product of the derivatives of f() and g(). Using *f'(x)* to refer to the derivative, we can say that: ```f'(x) = f'(u) * g'(x) = 2*u * 2 = 2*(2*x) * 2 = 8*x```.
# 
# Let's check our calculation:

# In[6]:


# sympy let's us do symbolic differentiation (and much more!) in python
import sympy as sp
# we have to define our variables
x = sp.var('x')
# then we can request the derivative or any expression of that variable
pow(2*x,2).diff()


# The key insight is that the stacking of linear functions and non-linear activations we learnt about in the last section is simply defining a function of functions (of functions, of functions...). Each layer is taking the output of the previous layer's function, and using it as input into its function. Therefore, we can calculate the derivative at any layer by simply multiplying the gradients of that layer and all of its following layers together! This use of the chain rule to allow us to rapidly calculate the derivatives of our model at any layer is referred to as *back propagation*.
# 
# The good news is that you'll never have to worry about the details of this yourself, since libraries like Theano and Tensorflow (and therefore wrappers like Keras) provide *automatic differentiation* (or *AD*). ***TODO***

# ### Training multiple layers in Keras

# The code below will work on any model that contains dense layers; it's not just for this VGG model.
# 
# NB: Don't skip the step of fine-tuning just the final layer first, since otherwise you'll have one layer with random weights, which will cause the other layers to quickly move a long way from their optimized imagenet weights.

# In[253]:


layers = model.layers
# Get the index of the first dense layer...
first_dense_idx = [index for index,layer in enumerate(layers) if type(layer) is Dense][0]
# ...and set this and all subsequent layers to trainable
for layer in layers[first_dense_idx:]: layer.trainable=True


# Since we haven't changed our architecture, there's no need to re-compile the model - instead, we just set the learning rate. Since we're training more layers, and since we've already optimized the last layer, we should use a lower learning rate than previously.

# In[254]:


K.set_value(opt.lr, 0.01)
fit_model(model, batches, val_batches, 3)


# This is an extraordinarily powerful 5 lines of code. We have fine-tuned all of our dense layers to be optimized for our specific data set. This kind of technique has only become accessible in the last year or two - and we can already do it in just 5 lines of python!

# In[255]:


model.save_weights(model_path+'finetune2.h5')


# There's generally little room for improvement in training the convolutional layers, if you're using the model on natural images (as we are). However, there's no harm trying a few of the later conv layers, since it may give a slight improvement, and can't hurt (and we can always load the previous weights if the accuracy decreases).

# In[256]:


for layer in layers[12:]: layer.trainable=True
K.set_value(opt.lr, 0.001)


# In[257]:


fit_model(model, batches, val_batches, 4)


# In[259]:


model.save_weights(model_path+'finetune3.h5')


# You can always load the weights later and use the model to do whatever you need:

# In[ ]:


model.load_weights(model_path+'finetune2.h5')
model.evaluate_generator(get_batches(path+'valid', gen, False, batch_size*2), val_batches.n)

