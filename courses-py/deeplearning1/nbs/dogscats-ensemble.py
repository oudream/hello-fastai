
# coding: utf-8

# In[5]:


# from theano.sandbox import cuda
# cuda.use('gpu0')


# In[6]:


get_ipython().magic(u'matplotlib inline')
import utils; reload(utils)
from utils import *
from __future__ import division, print_function


# ## Setup

# In[3]:


path = "/eee/dogscats/"
model_path = path + 'models/'
if not os.path.exists(model_path): os.mkdir(model_path)

batch_size=64


# In[4]:


batches = get_batches(path+'train', shuffle=False, batch_size=batch_size)
val_batches = get_batches(path+'valid', shuffle=False, batch_size=batch_size)


# In[6]:


(val_classes, trn_classes, val_labels, trn_labels, 
    val_filenames, filenames, test_filenames) = get_classes(path)


# In this notebook we're going to create an ensemble of models and use their average as our predictions. For each ensemble, we're going to follow our usual fine-tuning steps:
# 
# 1) Create a model that retrains just the last layer
# 2) Add this to a model containing all VGG layers except the last layer
# 3) Fine-tune just the dense layers of this model (pre-computing the convolutional layers)
# 4) Add data augmentation, fine-tuning the dense layers without pre-computation.
# 
# So first, we need to create our VGG model and pre-compute the output of the conv layers:

# In[15]:


model = Vgg16().model
conv_layers,fc_layers = split_at(model, Convolution2D)


# In[16]:


conv_model = Sequential(conv_layers)


# In[17]:


val_features = conv_model.predict_generator(val_batches, val_batches.nb_sample)
trn_features = conv_model.predict_generator(batches, batches.nb_sample)


# In[33]:


save_array(model_path + 'train_convlayer_features.bc', trn_features)
save_array(model_path + 'valid_convlayer_features.bc', val_features)


# In the future we can just load these precomputed features:

# In[6]:


trn_features = load_array(model_path+'train_convlayer_features.bc')
val_features = load_array(model_path+'valid_convlayer_features.bc')


# We can also save some time by pre-computing the training and validation arrays with the image decoding and resizing already done:

# In[7]:


trn = get_data(path+'train')
val = get_data(path+'valid')


# In[8]:


save_array(model_path+'train_data.bc', trn)
save_array(model_path+'valid_data.bc', val)


# In the future we can just load these resized images:

# In[7]:


trn = load_array(model_path+'train_data.bc')
val = load_array(model_path+'valid_data.bc')


# Finally, we can precompute the output of all but the last dropout and dense layers, for creating the first stage of the model:

# In[19]:


model.pop()
model.pop()


# In[20]:


ll_val_feat = model.predict_generator(val_batches, val_batches.nb_sample)
ll_feat = model.predict_generator(batches, batches.nb_sample)


# In[21]:


save_array(model_path + 'train_ll_feat.bc', ll_feat)
save_array(model_path + 'valid_ll_feat.bc', ll_val_feat)


# In[8]:


ll_feat = load_array(model_path+ 'train_ll_feat.bc')
ll_val_feat = load_array(model_path + 'valid_ll_feat.bc')


# ...and let's also grab the test data, for when we need to submit:

# In[16]:


test = get_data(path+'test')
save_array(model_path+'test_data.bc', test)


# In[22]:


test = load_array(model_path+'test_data.bc')


# ## Last layer

# The functions automate creating a model that trains the last layer from scratch, and then adds those new layers on to the main model.

# In[9]:


def get_ll_layers():
    return [ 
        BatchNormalization(input_shape=(4096,)),
        Dropout(0.5),
        Dense(2, activation='softmax') 
        ]


# In[46]:


def train_last_layer(i):
    ll_layers = get_ll_layers()
    ll_model = Sequential(ll_layers)
    ll_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    ll_model.optimizer.lr=1e-5
    ll_model.fit(ll_feat, trn_labels, validation_data=(ll_val_feat, val_labels), nb_epoch=12)
    ll_model.optimizer.lr=1e-7
    ll_model.fit(ll_feat, trn_labels, validation_data=(ll_val_feat, val_labels), nb_epoch=1)
    ll_model.save_weights(model_path+'ll_bn' + i + '.h5')

    vgg = Vgg16()
    model = vgg.model
    model.pop(); model.pop(); model.pop()
    for layer in model.layers: layer.trainable=False
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    ll_layers = get_ll_layers()
    for layer in ll_layers: model.add(layer)
    for l1,l2 in zip(ll_model.layers, model.layers[-3:]):
        l2.set_weights(l1.get_weights())
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    model.save_weights(model_path+'bn' + i + '.h5')
    return model


# ## Dense model

# In[47]:


def get_conv_model(model):
    layers = model.layers
    last_conv_idx = [index for index,layer in enumerate(layers) 
                         if type(layer) is Convolution2D][-1]

    conv_layers = layers[:last_conv_idx+1]
    conv_model = Sequential(conv_layers)
    fc_layers = layers[last_conv_idx+1:]
    return conv_model, fc_layers, last_conv_idx


# In[48]:


def get_fc_layers(p, in_shape):
    return [
        MaxPooling2D(input_shape=in_shape),
        Flatten(),
        Dense(4096, activation='relu'),
        BatchNormalization(),
        Dropout(p),
        Dense(4096, activation='relu'),
        BatchNormalization(),
        Dropout(p),
        Dense(2, activation='softmax')
        ]


# In[49]:


def train_dense_layers(i, model):
    conv_model, fc_layers, last_conv_idx = get_conv_model(model)
    conv_shape = conv_model.output_shape[1:]
    fc_model = Sequential(get_fc_layers(0.5, conv_shape))
    for l1,l2 in zip(fc_model.layers, fc_layers): 
        weights = l2.get_weights()
        l1.set_weights(weights)
    fc_model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', 
                     metrics=['accuracy'])
    fc_model.fit(trn_features, trn_labels, nb_epoch=2, 
         batch_size=batch_size, validation_data=(val_features, val_labels))

    gen = image.ImageDataGenerator(rotation_range=10, width_shift_range=0.05, 
       width_zoom_range=0.05, zoom_range=0.05,
       channel_shift_range=10, height_shift_range=0.05, shear_range=0.05, horizontal_flip=True)
    batches = gen.flow(trn, trn_labels, batch_size=batch_size)
    val_batches = image.ImageDataGenerator().flow(val, val_labels, 
                      shuffle=False, batch_size=batch_size)

    for layer in conv_model.layers: layer.trainable = False
    for layer in get_fc_layers(0.5, conv_shape): conv_model.add(layer)
    for l1,l2 in zip(conv_model.layers[last_conv_idx+1:], fc_model.layers): 
        l1.set_weights(l2.get_weights())

    conv_model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', 
                       metrics=['accuracy'])
    conv_model.save_weights(model_path+'no_dropout_bn' + i + '.h5')
    conv_model.fit_generator(batches, samples_per_epoch=batches.N, nb_epoch=1, 
                            validation_data=val_batches, nb_val_samples=val_batches.N)
    for layer in conv_model.layers[16:]: layer.trainable = True
    conv_model.fit_generator(batches, samples_per_epoch=batches.N, nb_epoch=8, 
                            validation_data=val_batches, nb_val_samples=val_batches.N)

    conv_model.optimizer.lr = 1e-7
    conv_model.fit_generator(batches, samples_per_epoch=batches.N, nb_epoch=10, 
                            validation_data=val_batches, nb_val_samples=val_batches.N)
    conv_model.save_weights(model_path + 'aug' + i + '.h5')


# ## Build ensemble

# In[50]:


for i in range(5):
    i = str(i)
    model = train_last_layer(i)
    train_dense_layers(i, model)


# ## Combine ensemble and test

# In[4]:


ens_model = vgg_ft(2)
for layer in ens_model.layers: layer.trainable=True


# In[52]:


def get_ens_pred(arr, fname):
    ens_pred = []
    for i in range(5):
        i = str(i)
        ens_model.load_weights('{}{}{}.h5'.format(model_path, fname, i))
        preds = ens_model.predict(arr, batch_size=batch_size)
        ens_pred.append(preds)
    return ens_pred


# In[55]:


val_pred2 = get_ens_pred(val, 'aug')


# In[56]:


val_avg_preds2 = np.stack(val_pred2).mean(axis=0)


# In[61]:


categorical_accuracy(val_labels, val_avg_preds2).eval()

