
# coding: utf-8

# # Training a better model

# In[1]:


from theano.sandbox import cuda


# In[2]:


get_ipython().magic(u'matplotlib inline')
import utils; reload(utils)
from utils import *
from __future__ import division, print_function


# In[3]:


#path = "data/dogscats/sample/"
path = "data/dogscats/"
model_path = path + 'models/'
if not os.path.exists(model_path): os.mkdir(model_path)

batch_size=64


# ## Are we underfitting?

# Our validation accuracy so far has generally been higher than our training accuracy. That leads to two obvious questions:
# 
# 1. How is this possible?
# 2. Is this desirable?
# 
# The answer to (1) is that this is happening because of *dropout*. Dropout refers to a layer that randomly deletes (i.e. sets to zero) each activation in the previous layer with probability *p* (generally 0.5). This only happens during training, not when calculating the accuracy on the validation set, which is why the validation set can show higher accuracy than the training set.
# 
# The purpose of dropout is to avoid overfitting. By deleting parts of the neural network at random during training, it ensures that no one part of the network can overfit to one part of the training set. The creation of dropout was one of the key developments in deep learning, and has allowed us to create rich models without overfitting. However, it can also result in underfitting if overused, and this is something we should be careful of with our model.
# 
# So the answer to (2) is: this is probably not desirable. It is likely that we can get better validation set results with less (or no) dropout, if we're seeing that validation accuracy is higher than training accuracy - a strong sign of underfitting. So let's try removing dropout entirely, and see what happens!
# 
# (We had dropout in this model already because the VGG authors found it necessary for the imagenet competition. But that doesn't mean it's necessary for dogs v cats, so we will do our own analysis of regularization approaches from scratch.)

# ## Removing dropout

# Our high level approach here will be to start with our fine-tuned cats vs dogs model (with dropout), then fine-tune all the dense layers, after removing dropout from them. The steps we will take are:
# - Re-create and load our modified VGG model with binary dependent (i.e. dogs v cats)
# - Split the model between the convolutional (*conv*) layers and the dense layers
# - Pre-calculate the output of the conv layers, so that we don't have to redundently re-calculate them on every epoch
# - Create a new model with just the dense layers, and dropout p set to zero
# - Train this new model using the output of the conv layers as training data.

# As before we need to start with a working model, so let's bring in our working VGG 16 model and change it to predict our binary dependent...

# In[4]:


model = vgg_ft(2)


# ...and load our fine-tuned weights.

# In[5]:


model.load_weights(model_path+'finetune3.h5')


# We're going to be training a number of iterations without dropout, so it would be best for us to pre-calculate the input to the fully connected layers - i.e. the *Flatten()* layer. We'll start by finding this layer in our model, and creating a new model that contains just the layers up to and including this layer:

# In[6]:


layers = model.layers


# In[7]:


last_conv_idx = [index for index,layer in enumerate(layers) 
                     if type(layer) is Convolution2D][-1]


# In[8]:


last_conv_idx


# In[9]:


layers[last_conv_idx]


# In[10]:


conv_layers = layers[:last_conv_idx+1]
conv_model = Sequential(conv_layers)
# Dense layers - also known as fully connected or 'FC' layers
fc_layers = layers[last_conv_idx+1:]


# Now we can use the exact same approach to creating features as we used when we created the linear model from the imagenet predictions in the last lesson - it's only the model that has changed. As you're seeing, there's a fairly small number of "recipes" that can get us a long way!

# In[4]:


batches = get_batches(path+'train', shuffle=False, batch_size=batch_size)
val_batches = get_batches(path+'valid', shuffle=False, batch_size=batch_size)

val_classes = val_batches.classes
trn_classes = batches.classes
val_labels = onehot(val_classes)
trn_labels = onehot(trn_classes)


# In[5]:


batches.class_indices


# In[12]:


val_features = conv_model.predict_generator(val_batches, val_batches.nb_sample)


# In[ ]:


trn_features = conv_model.predict_generator(batches, batches.nb_sample)


# In[41]:


save_array(model_path + 'train_convlayer_features.bc', trn_features)
save_array(model_path + 'valid_convlayer_features.bc', val_features)


# In[89]:


trn_features = load_array(model_path+'train_convlayer_features.bc')
val_features = load_array(model_path+'valid_convlayer_features.bc')


# In[90]:


trn_features.shape


# For our new fully connected model, we'll create it using the exact same architecture as the last layers of VGG 16, so that we can conveniently copy pre-trained weights over from that model. However, we'll set the dropout layer's p values to zero, so as to effectively remove dropout.

# In[16]:


# Copy the weights from the pre-trained model.
# NB: Since we're removing dropout, we want to half the weights
def proc_wgts(layer): return [o/2 for o in layer.get_weights()]


# In[17]:


# Such a finely tuned model needs to be updated very slowly!
opt = RMSprop(lr=0.00001, rho=0.7)


# In[18]:


def get_fc_model():
    model = Sequential([
        MaxPooling2D(input_shape=conv_layers[-1].output_shape[1:]),
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.),
        Dense(4096, activation='relu'),
        Dropout(0.),
        Dense(2, activation='softmax')
        ])

    for l1,l2 in zip(model.layers, fc_layers): l1.set_weights(proc_wgts(l2))

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# In[19]:


fc_model = get_fc_model()


# And fit the model in the usual way:

# In[15]:


fc_model.fit(trn_features, trn_labels, nb_epoch=8, 
             batch_size=batch_size, validation_data=(val_features, val_labels))


# In[16]:


fc_model.save_weights(model_path+'no_dropout.h5')


# In[ ]:


fc_model.load_weights(model_path+'no_dropout.h5')


# # Reducing overfitting

# Now that we've gotten the model to overfit, we can take a number of steps to reduce this.

# ## Approaches to reducing overfitting

# We do not necessarily need to rely on dropout or other regularization approaches to reduce overfitting. There are other techniques we should try first, since regularlization, by definition, biases our model towards simplicity - which we only want to do if we know that's necessary. This is the order that we recommend using for reducing overfitting (more details about each in a moment):
# 
# 1. Add more data
# 2. Use data augmentation
# 3. Use architectures that generalize well
# 4. Add regularization
# 5. Reduce architecture complexity.
# 
# We'll assume that you've already collected as much data as you can, so step (1) isn't relevant (this is true for most Kaggle competitions, for instance). So the next step (2) is data augmentation. This refers to creating additional synthetic data, based on reasonable modifications of your input data. For images, this is likely to involve one or more of: flipping, rotation, zooming, cropping, panning, minor color changes.
# 
# Which types of augmentation are appropriate depends on your data. For regular photos, for instance, you'll want to use horizontal flipping, but not vertical flipping (since an upside down car is much less common than a car the right way up, for instance!)
# 
# We recommend *always* using at least some light data augmentation, unless you have so much data that your model will never see the same input twice.

# ## About data augmentation

# Keras comes with very convenient features for automating data augmentation. You simply define what types and maximum amounts of augmentation you want, and keras ensures that every item of every batch randomly is changed according to these settings. Here's how to define a generator that includes data augmentation:

# In[26]:


# dim_ordering='tf' uses tensorflow dimension ordering,
#   which is the same order as matplotlib uses for display.
# Therefore when just using for display purposes, this is more convenient
gen = image.ImageDataGenerator(rotation_range=10, width_shift_range=0.1, 
       height_shift_range=0.1, shear_range=0.15, zoom_range=0.1, 
       channel_shift_range=10., horizontal_flip=True, dim_ordering='tf')


# Let's take a look at how this generator changes a single image (the details of this code don't matter much, but feel free to read the comments and keras docs to understand the details if you're interested).

# In[27]:


# Create a 'batch' of a single image
img = np.expand_dims(ndimage.imread('data/dogscats/test/7.jpg'),0)
# Request the generator to create batches from this image
aug_iter = gen.flow(img)


# In[28]:


# Get eight examples of these augmented images
aug_imgs = [next(aug_iter)[0].astype(np.uint8) for i in range(8)]


# In[12]:


# The original
plt.imshow(img[0])


# As you can see below, there's no magic to data augmentation - it's a very intuitive approach to generating richer input data. Generally speaking, your intuition should be a good guide to appropriate data augmentation, although it's a good idea to test your intuition by checking the results of different augmentation approaches.

# In[29]:


# Augmented data
plots(aug_imgs, (20,7), 2)


# In[22]:


# Ensure that we return to theano dimension ordering
K.set_image_dim_ordering('th')


# ## Adding data augmentation

# Let's try adding a small amount of data augmentation, and see if we reduce overfitting as a result. The approach will be identical to the method we used to finetune the dense layers in lesson 2, except that we will use a generator with augmentation configured. Here's how we set up the generator, and create batches from it:

# In[14]:


gen = image.ImageDataGenerator(rotation_range=15, width_shift_range=0.1, 
                               height_shift_range=0.1, zoom_range=0.1, horizontal_flip=True)


# In[15]:


batches = get_batches(path+'train', gen, batch_size=batch_size)
# NB: We don't want to augment or shuffle the validation set
val_batches = get_batches(path+'valid', shuffle=False, batch_size=batch_size)


# When using data augmentation, we can't pre-compute our convolutional layer features, since randomized changes are being made to every input image. That is, even if the training process sees the same image multiple times, each time it will have undergone different data augmentation, so the results of the convolutional layers will be different.
# 
# Therefore, in order to allow data to flow through all the conv layers and our new dense layers, we attach our fully connected model to the convolutional model--after ensuring that the convolutional layers are not trainable:

# In[18]:


fc_model = get_fc_model()


# In[19]:


for layer in conv_model.layers: layer.trainable = False
# Look how easy it is to connect two models together!
conv_model.add(fc_model)


# Now we can compile, train, and save our model as usual - note that we use *fit_generator()* since we want to pull random images from the directories on every batch.

# In[22]:


conv_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


# In[52]:


conv_model.fit_generator(batches, samples_per_epoch=batches.nb_sample, nb_epoch=8, 
                        validation_data=val_batches, nb_val_samples=val_batches.nb_sample)


# In[24]:


conv_model.fit_generator(batches, samples_per_epoch=batches.nb_sample, nb_epoch=3, 
                        validation_data=val_batches, nb_val_samples=val_batches.nb_sample)


# In[53]:


conv_model.save_weights(model_path + 'aug1.h5')


# In[23]:


conv_model.load_weights(model_path + 'aug1.h5')


# ## Batch normalization

# ### About batch normalization

# Batch normalization (*batchnorm*) is a way to ensure that activations don't become too high or too low at any point in the model. Adjusting activations so they are of similar scales is called *normalization*. Normalization is very helpful for fast training - if some activations are very high, they will saturate the model and create very large gradients, causing training to fail; if very low, they will cause training to proceed very slowly. Furthermore, large or small activations in one layer will tend to result in even larger or smaller activations in later layers, since the activations get multiplied repeatedly across the layers.
# 
# Prior to the development of batchnorm in 2015, only the inputs to a model could be effectively normalized - by simply subtracting their mean and dividing by their standard deviation. However, weights in intermediate layers could easily become poorly scaled, due to problems in weight initialization, or a high learning rate combined with random fluctuations in weights.

# Batchnorm resolves this problem by normalizing each intermediate layer as well. The details of how it works are not terribly important (although I will outline them in a moment) - the important takeaway is that **all modern networks should use batchnorm, or something equivalent**. There are two reasons for this:
# 1. Adding batchnorm to a model can result in **10x or more improvements in training speed**
# 2. Because normalization greatly reduces the ability of a small number of outlying inputs to over-influence the training, it also tends to **reduce overfitting**.

# As promised, here's a brief outline of how batchnorm works. As a first step, it normalizes intermediate layers in the same way as input layers can be normalized. But this on its own would not be enough, since the model would then just push the weights up or down indefinitely to try to undo this normalization. Therefore, batchnorm takes two additional steps:
# 1. Add two more trainable parameters to each layer - one to multiply all activations to set an arbitrary standard deviation, and one to add to all activations to set an arbitary mean
# 2. Incorporate both the normalization, and the learnt multiply/add parameters, into the gradient calculations during backprop.
# 
# This ensures that the weights don't tend to push very high or very low (since the normalization is included in the gradient calculations, so the updates are aware of the normalization). But it also ensures that if a layer does need to change the overall mean or standard deviation in order to match the output scale, it can do so.

# ### Adding batchnorm to the model

# We can use nearly the same approach as before - but this time we'll add batchnorm layers (and dropout layers):

# In[16]:


conv_layers[-1].output_shape[1:]


# In[34]:


def get_bn_layers(p):
    return [
        MaxPooling2D(input_shape=conv_layers[-1].output_shape[1:]),
        Flatten(),
        Dense(4096, activation='relu'),
        BatchNormalization(),
        Dropout(p),
        Dense(4096, activation='relu'),
        BatchNormalization(),
        Dropout(p),
        Dense(1000, activation='softmax')
        ]


# In[26]:


def load_fc_weights_from_vgg16bn(model):
    "Load weights for model from the dense layers of the Vgg16BN model."
    # See imagenet_batchnorm.ipynb for info on how the weights for
    # Vgg16BN can be generated from the standard Vgg16 weights.
    from vgg16bn import Vgg16BN
    vgg16_bn = Vgg16BN()
    _, fc_layers = split_at(vgg16_bn.model, Convolution2D)
    copy_weights(fc_layers, model.layers)


# In[27]:


p=0.6


# In[69]:


bn_model = Sequential(get_bn_layers(0.6))


# In[ ]:


load_fc_weights_from_vgg16bn(bn_model)


# In[71]:


def proc_wgts(layer, prev_p, new_p):
    scal = (1-prev_p)/(1-new_p)
    return [o*scal for o in layer.get_weights()]


# In[72]:


for l in bn_model.layers: 
    if type(l)==Dense: l.set_weights(proc_wgts(l, 0.5, 0.6))


# In[73]:


bn_model.pop()
for layer in bn_model.layers: layer.trainable=False


# In[74]:


bn_model.add(Dense(2,activation='softmax'))


# In[75]:


bn_model.compile(Adam(), 'categorical_crossentropy', metrics=['accuracy'])


# In[76]:


bn_model.fit(trn_features, trn_labels, nb_epoch=8, validation_data=(val_features, val_labels))


# In[77]:


bn_model.save_weights(model_path+'bn.h5')


# In[44]:


bn_model.load_weights(model_path+'bn.h5')


# In[78]:


bn_layers = get_bn_layers(0.6)
bn_layers.pop()
bn_layers.append(Dense(2,activation='softmax'))


# In[79]:


final_model = Sequential(conv_layers)
for layer in final_model.layers: layer.trainable = False
for layer in bn_layers: final_model.add(layer)


# In[80]:


for l1,l2 in zip(bn_model.layers, bn_layers):
    l2.set_weights(l1.get_weights())


# In[81]:


final_model.compile(optimizer=Adam(), 
                    loss='categorical_crossentropy', metrics=['accuracy'])


# In[82]:


final_model.fit_generator(batches, samples_per_epoch=batches.nb_sample, nb_epoch=1, 
                        validation_data=val_batches, nb_val_samples=val_batches.nb_sample)


# In[83]:


final_model.save_weights(model_path + 'final1.h5')


# In[84]:


final_model.fit_generator(batches, samples_per_epoch=batches.nb_sample, nb_epoch=4, 
                        validation_data=val_batches, nb_val_samples=val_batches.nb_sample)


# In[85]:


final_model.save_weights(model_path + 'final2.h5')


# In[86]:


final_model.optimizer.lr=0.001


# In[92]:


final_model.fit_generator(batches, samples_per_epoch=batches.nb_sample, nb_epoch=4, 
                        validation_data=val_batches, nb_val_samples=val_batches.nb_sample)


# In[ ]:


bn_model.save_weights(model_path + 'final3.h5')

