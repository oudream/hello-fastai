
# coding: utf-8

# This notebook explains how to add batch normalization to VGG.  The code shown here is implemented in [vgg_bn.py](https://github.com/fastai/courses/blob/master/deeplearning1/nbs/vgg16bn.py), and there is a version of ``vgg_ft`` (our fine tuning function) with batch norm called ``vgg_ft_bn`` in [utils.py](https://github.com/fastai/courses/blob/master/deeplearning1/nbs/utils.py).

# In[1]:


from theano.sandbox import cuda


# In[2]:


get_ipython().magic(u'matplotlib inline')
import utils; reload(utils)
from utils import *
from __future__ import print_function, division


# # The problem, and the solution

# ## The problem

# The problem that we faced in the lesson 3 is that when we wanted to add batch normalization, we initialized *all* the dense layers of the model to random weights, and then tried to train them with our cats v dogs dataset. But that's a lot of weights to initialize to random - out of 134m params, around 119m are in the dense layers! Take a moment to think about why this is, and convince yourself that dense layers are where most of the weights will be. Also, think about whether this implies that most of the *time* will be spent training these weights. What do you think?
# 
# Trying to train 120m params using just 23k images is clearly an unreasonable expectation. The reason we haven't had this problem before is that the dense layers were not random, but were trained to recognize imagenet categories (other than the very last layer, which only has 8194 params).

# ## The solution

# The solution, obviously enough, is to add batch normalization to the VGG model! To do so, we have to be careful - we can't just insert batchnorm layers, since their parameters (*gamma* - which is used to multiply by each activation, and *beta* - which is used to add to each activation) will not be set correctly. Without setting these correctly, the new batchnorm layers will normalize the previous layer's activations, meaning that the next layer will receive totally different activations to what it would have without new batchnorm layer. And that means that all the pre-trained weights are no longer of any use!
# 
# So instead, we need to figure out what beta and gamma to choose when we insert the layers. The answer to this turns out to be pretty simple - we need to calculate what the mean and standard deviation of that activations for that layer are when calculated on all of imagenet, and then set beta and gamma to these values. That means that the new batchnorm layer will normalize the data with the mean and standard deviation, and then immediately un-normalize the data using the beta and gamma parameters we provide. So the output of the batchnorm layer will be identical to it's input - which means that all the pre-trained weights will continue to work just as well as before.
# 
# The benefit of this is that when we wish to fine-tune our own networks, we will have all the benefits of batch normalization (higher learning rates, more resiliant training, and less need for dropout) plus all the benefits of a pre-trained network.

# To calculate the mean and standard deviation of the activations on imagenet, we need to download imagenet. You can download imagenet from http://www.image-net.org/download-images . The file you want is the one titled **Download links to ILSVRC2013 image data**. You'll need to request access from the imagenet admins for this, although it seems to be an automated system - I've always found that access is provided instantly. Once you're logged in and have gone to that page, look for the **CLS-LOC dataset** section. Both training and validation images are available, and you should download both. There's not much reason to download the test images, however.
# 
# Note that this will not be the entire imagenet archive, but just the 1000 categories that are used in the annual competition. Since that's what VGG16 was originally trained on, that seems like a good choice - especially since the full dataset is 1.1 terabytes, whereas the 1000 category dataset is 138 gigabytes.

# # Adding batchnorm to Imagenet

# ## Setup

# ### Sample

# As per usual, we create a sample so we can experiment more rapidly.

# In[ ]:


get_ipython().magic(u'pushd data/imagenet')
get_ipython().magic(u'cd train')


# In[6]:


get_ipython().magic(u'mkdir ../sample')
get_ipython().magic(u'mkdir ../sample/train')
get_ipython().magic(u'mkdir ../sample/valid')

from shutil import copyfile

g = glob('*')
for d in g: 
    os.mkdir('../sample/train/'+d)
    os.mkdir('../sample/valid/'+d)


# In[8]:


g = glob('*/*.JPEG')
shuf = np.random.permutation(g)
for i in range(25000): copyfile(shuf[i], '../sample/train/' + shuf[i])


# In[10]:


get_ipython().magic(u'cd ../valid')

g = glob('*/*.JPEG')
shuf = np.random.permutation(g)
for i in range(5000): copyfile(shuf[i], '../sample/valid/' + shuf[i])

get_ipython().magic(u'cd ..')


# In[11]:


get_ipython().magic(u'mkdir sample/results')


# In[ ]:


get_ipython().magic(u'popd')


# ### Data setup

# We set up our paths, data, and labels in the usual way. Note that we don't try to read all of Imagenet into memory! We only load the sample into memory.

# In[2]:


sample_path = 'data/jhoward/imagenet/sample/'
# This is the path to my fast SSD - I put datasets there when I can to get the speed benefit
fast_path = '/home/jhoward/ILSVRC2012_img_proc/'
#path = '/data/jhoward/imagenet/sample/'
path = 'data/jhoward/imagenet/'


# In[3]:


batch_size=64


# In[9]:


samp_trn = get_data(path+'train')
samp_val = get_data(path+'valid')


# In[10]:


save_array(samp_path+'results/trn.dat', samp_trn)
save_array(samp_path+'results/val.dat', samp_val)


# In[ ]:


samp_trn = load_array(sample_path+'results/trn.dat')
samp_val = load_array(sample_path+'results/val.dat')


# In[5]:


(val_classes, trn_classes, val_labels, trn_labels, 
    val_filenames, filenames, test_filenames) = get_classes(path)


# In[58]:


(samp_val_classes, samp_trn_classes, samp_val_labels, samp_trn_labels, 
    samp_val_filenames, samp_filenames, samp_test_filenames) = get_classes(sample_path)


# ### Model setup

# Since we're just working with the dense layers, we should pre-compute the output of the convolutional layers.

# In[4]:


vgg = Vgg16()
model = vgg.model


# In[5]:


layers = model.layers
last_conv_idx = [index for index,layer in enumerate(layers) 
                     if type(layer) is Convolution2D][-1]
conv_layers = layers[:last_conv_idx+1]


# In[6]:


dense_layers = layers[last_conv_idx+1:]


# In[7]:


conv_model = Sequential(conv_layers)


# In[68]:


samp_conv_val_feat = conv_model.predict(samp_val, batch_size=batch_size*2)
samp_conv_feat = conv_model.predict(samp_trn, batch_size=batch_size*2)


# In[70]:


save_array(sample_path+'results/conv_val_feat.dat', samp_conv_val_feat)
save_array(sample_path+'results/conv_feat.dat', samp_conv_feat)


# In[9]:


samp_conv_feat = load_array(sample_path+'results/conv_feat.dat')
samp_conv_val_feat = load_array(sample_path+'results/conv_val_feat.dat')


# In[10]:


samp_conv_val_feat.shape


# This is our usual Vgg network just covering the dense layers:

# In[ ]:


def get_dense_layers():
    return [
        MaxPooling2D(input_shape=conv_layers[-1].output_shape[1:]),
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(1000, activation='softmax')
        ]


# In[ ]:


dense_model = Sequential(get_dense_layers())


# In[ ]:


for l1, l2 in zip(dense_layers, dense_model.layers):
    l2.set_weights(l1.get_weights())


# ### Check model

# It's a good idea to check that your models are giving reasonable answers, before using them.

# In[75]:


dense_model.compile(Adam(), 'categorical_crossentropy', ['accuracy'])


# In[76]:


dense_model.evaluate(samp_conv_val_feat, samp_val_labels)


# In[24]:


model.compile(Adam(), 'categorical_crossentropy', ['accuracy'])


# In[25]:


# should be identical to above
model.evaluate(val, val_labels)


# In[26]:


# should be a little better than above, since VGG authors overfit
dense_model.evaluate(conv_feat, trn_labels)


# ## Adding our new layers

# ### Calculating batchnorm params

# To calculate the output of a layer in a Keras sequential model, we have to create a function that defines the input layer and the output layer, like this:

# In[14]:


k_layer_out = K.function([dense_model.layers[0].input, K.learning_phase()], 
                         [dense_model.layers[2].output])


# Then we can call the function to get our layer activations:

# In[15]:


d0_out = k_layer_out([samp_conv_val_feat, 0])[0]


# In[16]:


k_layer_out = K.function([dense_model.layers[0].input, K.learning_phase()], 
                         [dense_model.layers[4].output])


# In[17]:


d2_out = k_layer_out([samp_conv_val_feat, 0])[0]


# Now that we've got our activations, we can calculate the mean and standard deviation for each (note that due to a bug in keras, it's actually the variance that we'll need).

# In[18]:


mu0,var0 = d0_out.mean(axis=0), d0_out.var(axis=0)
mu2,var2 = d2_out.mean(axis=0), d2_out.var(axis=0)


# ### Creating batchnorm model

# Now we're ready to create and insert our layers just after each dense layer.

# In[19]:


nl1 = BatchNormalization()
nl2 = BatchNormalization()


# In[20]:


bn_model = insert_layer(dense_model, nl2, 5)
bn_model = insert_layer(bn_model, nl1, 3)


# In[22]:


bnl1 = bn_model.layers[3]
bnl4 = bn_model.layers[6]


# After inserting the layers, we can set their weights to the variance and mean we just calculated.

# In[23]:


bnl1.set_weights([var0, mu0, mu0, var0])
bnl4.set_weights([var2, mu2, mu2, var2])


# In[21]:


bn_model.compile(Adam(1e-5), 'categorical_crossentropy', ['accuracy'])


# We should find that the new model gives identical results to those provided by the original VGG model.

# In[24]:


bn_model.evaluate(samp_conv_val_feat, samp_val_labels)


# In[25]:


bn_model.evaluate(samp_conv_feat, samp_trn_labels)


# ### Optional - additional fine-tuning

# Now that we have a VGG model with batchnorm, we might expect that the optimal weights would be a little different to what they were when originally created without batchnorm. So we fine tune the weights for one epoch.

# In[26]:


feat_bc = bcolz.open(fast_path+'trn_features.dat')


# In[27]:


labels = load_array(fast_path+'trn_labels.dat')


# In[28]:


val_feat_bc = bcolz.open(fast_path+'val_features.dat')


# In[29]:


val_labels = load_array(fast_path+'val_labels.dat')


# In[35]:


bn_model.fit(feat_bc, labels, nb_epoch=1, batch_size=batch_size,
             validation_data=(val_feat_bc, val_labels))


# The results look quite encouraging! Note that these VGG weights are now specific to how keras handles image scaling - that is, it squashes and stretches images, rather than adding black borders. So this model is best used on images created in that way.

# In[36]:


bn_model.save_weights(path+'models/bn_model2.h5')


# In[40]:


bn_model.load_weights(path+'models/bn_model2.h5')


# ### Create combined model

# Our last step is simply to copy our new dense layers on to the end of the convolutional part of the network, and save the new complete set of weights, so we can use them in the future when using VGG. (Of course, we'll also need to update our VGG architecture to add the batchnorm layers).

# In[54]:


new_layers = copy_layers(bn_model.layers)
for layer in new_layers:
    conv_model.add(layer)


# In[56]:


copy_weights(bn_model.layers, new_layers)


# In[63]:


conv_model.compile(Adam(1e-5), 'categorical_crossentropy', ['accuracy'])


# In[65]:


conv_model.evaluate(samp_val, samp_val_labels)


# In[66]:


conv_model.save_weights(path+'models/inet_224squash_bn.h5')


# The code shown here is implemented in [vgg_bn.py](https://github.com/fastai/courses/blob/master/deeplearning1/nbs/vgg16bn.py), and there is a version of ``vgg_ft`` (our fine tuning function) with batch norm called ``vgg_ft_bn`` in [utils.py](https://github.com/fastai/courses/blob/master/deeplearning1/nbs/utils.py).
