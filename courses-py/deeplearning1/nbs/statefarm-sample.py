
# coding: utf-8

# # Enter State Farm

# In[1]:


from theano.sandbox import cuda
cuda.use('gpu1')


# In[2]:


get_ipython().magic(u'matplotlib inline')
from __future__ import print_function, division
#path = "data/state/"
path = "data/state/sample/"
import utils; reload(utils)
from utils import *
from IPython.display import FileLink


# In[3]:


batch_size=64


# ## Create sample

# The following assumes you've already created your validation set - remember that the training and validation set should contain *different drivers*, as mentioned on the Kaggle competition page.

# In[ ]:


get_ipython().magic(u'cd data/state')


# In[ ]:


get_ipython().magic(u'cd train')


# In[ ]:


get_ipython().magic(u'mkdir ../sample')
get_ipython().magic(u'mkdir ../sample/train')
get_ipython().magic(u'mkdir ../sample/valid')


# In[ ]:


for d in glob('c?'):
    os.mkdir('../sample/train/'+d)
    os.mkdir('../sample/valid/'+d)


# In[ ]:


from shutil import copyfile


# In[ ]:


g = glob('c?/*.jpg')
shuf = np.random.permutation(g)
for i in range(1500): copyfile(shuf[i], '../sample/train/' + shuf[i])


# In[ ]:


get_ipython().magic(u'cd ../valid')


# In[ ]:


g = glob('c?/*.jpg')
shuf = np.random.permutation(g)
for i in range(1000): copyfile(shuf[i], '../sample/valid/' + shuf[i])


# In[ ]:


get_ipython().magic(u'cd ../../..')


# In[ ]:


get_ipython().magic(u'mkdir data/state/results')


# In[8]:


get_ipython().magic(u'mkdir data/state/sample/test')


# ## Create batches

# In[56]:


batches = get_batches(path+'train', batch_size=batch_size)
val_batches = get_batches(path+'valid', batch_size=batch_size*2, shuffle=False)


# In[5]:


(val_classes, trn_classes, val_labels, trn_labels, val_filenames, filenames,
    test_filename) = get_classes(path)


# ## Basic models

# ### Linear model

# First, we try the simplest model and use default parameters. Note the trick of making the first layer a batchnorm layer - that way we don't have to worry about normalizing the input ourselves.

# In[6]:


model = Sequential([
        BatchNormalization(axis=1, input_shape=(3,224,224)),
        Flatten(),
        Dense(10, activation='softmax')
    ])


# As you can see below, this training is going nowhere...

# In[7]:


model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(batches, batches.nb_sample, nb_epoch=2, validation_data=val_batches, 
                 nb_val_samples=val_batches.nb_sample)


# Let's first check the number of parameters to see that there's enough parameters to find some useful relationships:

# In[66]:


model.summary()


# Over 1.5 million parameters - that should be enough. Incidentally, it's worth checking you understand why this is the number of parameters in this layer:

# In[67]:


10*3*224*224


# Since we have a simple model with no regularization and plenty of parameters, it seems most likely that our learning rate is too high. Perhaps it is jumping to a solution where it predicts one or two classes with high confidence, so that it can give a zero prediction to as many classes as possible - that's the best approach for a model that is no better than random, and there is likely to be where we would end up with a high learning rate. So let's check:

# In[10]:


np.round(model.predict_generator(batches, batches.N)[:10],2)


# Our hypothesis was correct. It's nearly always predicting class 1 or 6, with very high confidence. So let's try a lower learning rate:

# In[14]:


model = Sequential([
        BatchNormalization(axis=1, input_shape=(3,224,224)),
        Flatten(),
        Dense(10, activation='softmax')
    ])
model.compile(Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(batches, batches.nb_sample, nb_epoch=2, validation_data=val_batches, 
                 nb_val_samples=val_batches.nb_sample)


# Great - we found our way out of that hole... Now we can increase the learning rate and see where we can get to.

# In[15]:


model.optimizer.lr=0.001


# In[16]:


model.fit_generator(batches, batches.nb_sample, nb_epoch=4, validation_data=val_batches, 
                 nb_val_samples=val_batches.nb_sample)


# We're stabilizing at validation accuracy of 0.39. Not great, but a lot better than random. Before moving on, let's check that our validation set on the sample is large enough that it gives consistent results:

# In[6]:


rnd_batches = get_batches(path+'valid', batch_size=batch_size*2, shuffle=True)


# In[11]:


val_res = [model.evaluate_generator(rnd_batches, rnd_batches.nb_sample) for i in range(10)]
np.round(val_res, 2)


# Yup, pretty consistent - if we see improvements of 3% or more, it's probably not random, based on the above samples.

# ### L2 regularization

# The previous model is over-fitting a lot, but we can't use dropout since we only have one layer. We can try to decrease overfitting in our model by adding [l2 regularization](http://www.kdnuggets.com/2015/04/preventing-overfitting-neural-networks.html/2) (i.e. add the sum of squares of the weights to our loss function):

# In[20]:


model = Sequential([
        BatchNormalization(axis=1, input_shape=(3,224,224)),
        Flatten(),
        Dense(10, activation='softmax', W_regularizer=l2(0.01))
    ])
model.compile(Adam(lr=10e-5), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(batches, batches.nb_sample, nb_epoch=2, validation_data=val_batches, 
                 nb_val_samples=val_batches.nb_sample)


# In[21]:


model.optimizer.lr=0.001


# In[22]:


model.fit_generator(batches, batches.nb_sample, nb_epoch=4, validation_data=val_batches, 
                 nb_val_samples=val_batches.nb_sample)


# Looks like we can get a bit over 50% accuracy this way. This will be a good benchmark for our future models - if we can't beat 50%, then we're not even beating a linear model trained on a sample, so we'll know that's not a good approach.

# ### Single hidden layer

# The next simplest model is to add a single hidden layer.

# In[34]:


model = Sequential([
        BatchNormalization(axis=1, input_shape=(3,224,224)),
        Flatten(),
        Dense(100, activation='relu'),
        BatchNormalization(),
        Dense(10, activation='softmax')
    ])
model.compile(Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(batches, batches.nb_sample, nb_epoch=2, validation_data=val_batches, 
                 nb_val_samples=val_batches.nb_sample)

model.optimizer.lr = 0.01
model.fit_generator(batches, batches.nb_sample, nb_epoch=5, validation_data=val_batches, 
                 nb_val_samples=val_batches.nb_sample)


# Not looking very encouraging... which isn't surprising since we know that CNNs are a much better choice for computer vision problems. So we'll try one.

# ### Single conv layer

# 2 conv layers with max pooling followed by a simple dense network is a good simple CNN to start with:

# In[61]:


def conv1(batches):
    model = Sequential([
            BatchNormalization(axis=1, input_shape=(3,224,224)),
            Convolution2D(32,3,3, activation='relu'),
            BatchNormalization(axis=1),
            MaxPooling2D((3,3)),
            Convolution2D(64,3,3, activation='relu'),
            BatchNormalization(axis=1),
            MaxPooling2D((3,3)),
            Flatten(),
            Dense(200, activation='relu'),
            BatchNormalization(),
            Dense(10, activation='softmax')
        ])

    model.compile(Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit_generator(batches, batches.nb_sample, nb_epoch=2, validation_data=val_batches, 
                     nb_val_samples=val_batches.nb_sample)
    model.optimizer.lr = 0.001
    model.fit_generator(batches, batches.nb_sample, nb_epoch=4, validation_data=val_batches, 
                     nb_val_samples=val_batches.nb_sample)
    return model


# In[62]:


conv1(batches)


# The training set here is very rapidly reaching a very high accuracy. So if we could regularize this, perhaps we could get a reasonable result.
# 
# So, what kind of regularization should we try first? As we discussed in lesson 3, we should start with data augmentation.

# ## Data augmentation

# To find the best data augmentation parameters, we can try each type of data augmentation, one at a time. For each type, we can try four very different levels of augmentation, and see which is the best. In the steps below we've only kept the single best result we found. We're using the CNN we defined above, since we have already observed it can model the data quickly and accurately.

# Width shift: move the image left and right -

# In[63]:


gen_t = image.ImageDataGenerator(width_shift_range=0.1)
batches = get_batches(path+'train', gen_t, batch_size=batch_size)


# In[64]:


model = conv1(batches)


# Height shift: move the image up and down -

# In[65]:


gen_t = image.ImageDataGenerator(height_shift_range=0.05)
batches = get_batches(path+'train', gen_t, batch_size=batch_size)


# In[66]:


model = conv1(batches)


# Random shear angles (max in radians) -

# In[67]:


gen_t = image.ImageDataGenerator(shear_range=0.1)
batches = get_batches(path+'train', gen_t, batch_size=batch_size)


# In[68]:


model = conv1(batches)


# Rotation: max in degrees -

# In[69]:


gen_t = image.ImageDataGenerator(rotation_range=15)
batches = get_batches(path+'train', gen_t, batch_size=batch_size)


# In[70]:


model = conv1(batches)


# Channel shift: randomly changing the R,G,B colors - 

# In[76]:


gen_t = image.ImageDataGenerator(channel_shift_range=20)
batches = get_batches(path+'train', gen_t, batch_size=batch_size)


# In[77]:


model = conv1(batches)


# And finally, putting it all together!

# In[75]:


gen_t = image.ImageDataGenerator(rotation_range=15, height_shift_range=0.05, 
                shear_range=0.1, channel_shift_range=20, width_shift_range=0.1)
batches = get_batches(path+'train', gen_t, batch_size=batch_size)


# In[59]:


model = conv1(batches)


# At first glance, this isn't looking encouraging, since the validation set is poor and getting worse. But the training set is getting better, and still has a long way to go in accuracy - so we should try annealing our learning rate and running more epochs, before we make a decisions.

# In[60]:


model.optimizer.lr = 0.0001
model.fit_generator(batches, batches.nb_sample, nb_epoch=5, validation_data=val_batches, 
                 nb_val_samples=val_batches.nb_sample)


# Lucky we tried that - we starting to make progress! Let's keep going.

# In[61]:


model.fit_generator(batches, batches.nb_sample, nb_epoch=25, validation_data=val_batches, 
                 nb_val_samples=val_batches.nb_sample)


# Amazingly, using nothing but a small sample, a simple (not pre-trained) model with no dropout, and data augmentation, we're getting results that would get us into the top 50% of the competition! This looks like a great foundation for our futher experiments.
# 
# To go further, we'll need to use the whole dataset, since dropout and data volumes are very related, so we can't tweak dropout without using all the data.
