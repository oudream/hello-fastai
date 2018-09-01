
# coding: utf-8

# # Fisheries competition

# In this notebook we're going to investigate a range of different architectures for the [Kaggle fisheries competition](https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/).  The video states that vgg.py and ``vgg_ft()`` from utils.py have been updated to include VGG with batch normalization, but this is not the case.  We've instead created a new file [vgg_bn.py](https://github.com/fastai/courses/blob/master/deeplearning1/nbs/vgg16bn.py) and an additional method ``vgg_ft_bn()`` (which is already in utils.py) which we use in this notebook.

# In[2]:


from theano.sandbox import cuda


# In[3]:


get_ipython().magic(u'matplotlib inline')
import utils; reload(utils)
from utils import *
from __future__ import division, print_function


# In[4]:


#path = "data/fish/sample/"
path = "data/fish/"
batch_size=64


# In[5]:


batches = get_batches(path+'train', batch_size=batch_size)
val_batches = get_batches(path+'valid', batch_size=batch_size*2, shuffle=False)

(val_classes, trn_classes, val_labels, trn_labels, 
    val_filenames, filenames, test_filenames) = get_classes(path)


# Sometimes it's helpful to have just the filenames, without the path.

# In[6]:


raw_filenames = [f.split('/')[-1] for f in filenames]
raw_test_filenames = [f.split('/')[-1] for f in test_filenames]
raw_val_filenames = [f.split('/')[-1] for f in val_filenames]


# ## Setup dirs

# We create the validation and sample sets in the usual way.

# In[ ]:


get_ipython().magic(u'cd data/fish')
get_ipython().magic(u'cd train')
get_ipython().magic(u'mkdir ../valid')


# In[ ]:


g = glob('*')
for d in g: os.mkdir('../valid/'+d)

g = glob('*/*.jpg')
shuf = np.random.permutation(g)
for i in range(500): os.rename(shuf[i], '../valid/' + shuf[i])


# In[ ]:


get_ipython().magic(u'mkdir ../sample')
get_ipython().magic(u'mkdir ../sample/train')
get_ipython().magic(u'mkdir ../sample/valid')


# In[ ]:


from shutil import copyfile

g = glob('*')
for d in g: 
    os.mkdir('../sample/train/'+d)
    os.mkdir('../sample/valid/'+d)


# In[ ]:


g = glob('*/*.jpg')
shuf = np.random.permutation(g)
for i in range(400): copyfile(shuf[i], '../sample/train/' + shuf[i])

get_ipython().magic(u'cd ../valid')

g = glob('*/*.jpg')
shuf = np.random.permutation(g)
for i in range(200): copyfile(shuf[i], '../sample/valid/' + shuf[i])

get_ipython().magic(u'cd ..')


# In[6]:


get_ipython().magic(u'mkdir results')
get_ipython().magic(u'mkdir sample/results')
get_ipython().magic(u'cd ../..')


# ## Basic VGG

# We start with our usual VGG approach.  We will be using VGG with batch normalization.  We explained how to add batch normalization to VGG in the [imagenet_batchnorm notebook](https://github.com/fastai/courses/blob/master/deeplearning1/nbs/imagenet_batchnorm.ipynb).  VGG with batch normalization is implemented in [vgg_bn.py](https://github.com/fastai/courses/blob/master/deeplearning1/nbs/vgg16bn.py), and there is a version of ``vgg_ft`` (our fine tuning function) with batch norm called ``vgg_ft_bn`` in [utils.py](https://github.com/fastai/courses/blob/master/deeplearning1/nbs/utils.py).

# ### Initial model

# First we create a simple fine-tuned VGG model to be our starting point.

# In[7]:


from vgg16bn import Vgg16BN
model = vgg_ft_bn(8)


# In[8]:


trn = get_data(path+'train')
val = get_data(path+'valid')


# In[9]:


test = get_data(path+'test')


# In[10]:


save_array(path+'results/trn.dat', trn)
save_array(path+'results/val.dat', val)


# In[11]:


save_array(path+'results/test.dat', test)


# In[45]:


trn = load_array(path+'results/trn.dat')
val = load_array(path+'results/val.dat')


# In[54]:


test = load_array(path+'results/test.dat')


# In[12]:


gen = image.ImageDataGenerator()


# In[13]:


model.compile(optimizer=Adam(1e-3),
       loss='categorical_crossentropy', metrics=['accuracy'])


# In[14]:


model.fit(trn, trn_labels, batch_size=batch_size, nb_epoch=3, validation_data=(val, val_labels))


# In[15]:


model.save_weights(path+'results/ft1.h5')


# ### Precompute convolutional output

# We pre-compute the output of the last convolution layer of VGG, since we're unlikely to need to fine-tune those layers. (All following analysis will be done on just the pre-computed convolutional features.)

# In[50]:


model.load_weights(path+'results/ft1.h5')


# In[16]:


conv_layers,fc_layers = split_at(model, Convolution2D)


# In[17]:


conv_model = Sequential(conv_layers)


# In[18]:


conv_feat = conv_model.predict(trn)
conv_val_feat = conv_model.predict(val)


# In[19]:


conv_test_feat = conv_model.predict(test)


# In[20]:


save_array(path+'results/conv_val_feat.dat', conv_val_feat)
save_array(path+'results/conv_feat.dat', conv_feat)


# In[21]:


save_array(path+'results/conv_test_feat.dat', conv_test_feat)


# In[53]:


conv_feat = load_array(path+'results/conv_feat.dat')
conv_val_feat = load_array(path+'results/conv_val_feat.dat')


# In[829]:


conv_test_feat = load_array(path+'results/conv_test_feat.dat')


# In[22]:


conv_val_feat.shape


# ### Train model

# We can now create our first baseline model - a simple 3-layer FC net.

# In[23]:


def get_bn_layers(p):
    return [
        MaxPooling2D(input_shape=conv_layers[-1].output_shape[1:]),
        BatchNormalization(axis=1),
        Dropout(p/4),
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(p),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(p/2),
        Dense(8, activation='softmax')
    ]


# In[24]:


p=0.6


# In[25]:


bn_model = Sequential(get_bn_layers(p))
bn_model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])


# In[26]:


bn_model.fit(conv_feat, trn_labels, batch_size=batch_size, nb_epoch=3, 
             validation_data=(conv_val_feat, val_labels))


# In[27]:


bn_model.optimizer.lr = 1e-4


# In[28]:


bn_model.fit(conv_feat, trn_labels, batch_size=batch_size, nb_epoch=7, 
             validation_data=(conv_val_feat, val_labels))


# In[30]:


bn_model.save_weights(path+'models/conv_512_6.h5')


# In[31]:


bn_model.evaluate(conv_val_feat, val_labels)


# In[774]:


bn_model.load_weights(path+'models/conv_512_6.h5')


# ## Multi-input

# The images are of different sizes, which are likely to represent the boat they came from (since different boats will use different cameras). Perhaps this creates some data leakage that we can take advantage of to get a better Kaggle leaderboard position? To find out, first we create arrays of the file sizes for each image:

# In[32]:


sizes = [PIL.Image.open(path+'train/'+f).size for f in filenames]
id2size = list(set(sizes))
size2id = {o:i for i,o in enumerate(id2size)}


# In[33]:


import collections
collections.Counter(sizes)


# Then we one-hot encode them (since we want to treat them as categorical) and normalize the data.

# In[34]:


trn_sizes_orig = to_categorical([size2id[o] for o in sizes], len(id2size))


# In[35]:


raw_val_sizes = [PIL.Image.open(path+'valid/'+f).size for f in val_filenames]
val_sizes = to_categorical([size2id[o] for o in raw_val_sizes], len(id2size))


# In[36]:


trn_sizes = trn_sizes_orig-trn_sizes_orig.mean(axis=0)/trn_sizes_orig.std(axis=0)
val_sizes = val_sizes-trn_sizes_orig.mean(axis=0)/trn_sizes_orig.std(axis=0)


# To use this additional "meta-data", we create a model with multiple input layers - `sz_inp` will be our input for the size information.

# In[37]:


p=0.6


# In[38]:


inp = Input(conv_layers[-1].output_shape[1:])
sz_inp = Input((len(id2size),))
bn_inp = BatchNormalization()(sz_inp)

x = MaxPooling2D()(inp)
x = BatchNormalization(axis=1)(x)
x = Dropout(p/4)(x)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(p)(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(p/2)(x)
x = merge([x,bn_inp], 'concat')
x = Dense(8, activation='softmax')(x)


# When we compile the model, we have to specify all the input layers in an array.

# In[39]:


model = Model([inp, sz_inp], x)
model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])


# And when we train the model, we have to provide all the input layers' data in an array.

# In[40]:


model.fit([conv_feat, trn_sizes], trn_labels, batch_size=batch_size, nb_epoch=3, 
             validation_data=([conv_val_feat, val_sizes], val_labels))


# In[41]:


bn_model.optimizer.lr = 1e-4


# In[42]:


bn_model.fit(conv_feat, trn_labels, batch_size=batch_size, nb_epoch=8, 
             validation_data=(conv_val_feat, val_labels))


# The model did not show an improvement by using the leakage, other than in the early epochs. This is most likely because the information about what boat the picture came from is readily identified from the image itself, so the meta-data turned out not to add any additional information.

# ## Bounding boxes & multi output

# ### Import / view bounding boxes

# A kaggle user has created bounding box annotations for each fish in each training set image. You can download them [from here](https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/forums/t/25902/complete-bounding-box-annotation). We will see if we can utilize this additional information. First, we'll load in the data, and keep just the largest bounding box for each image.

# In[44]:


import ujson as json


# In[45]:


anno_classes = ['alb', 'bet', 'dol', 'lag', 'other', 'shark', 'yft']


# In[3]:


def get_annotations():
    annot_urls = {
        '5458/bet_labels.json': 'bd20591439b650f44b36b72a98d3ce27',
        '5459/shark_labels.json': '94b1b3110ca58ff4788fb659eda7da90',
        '5460/dol_labels.json': '91a25d29a29b7e8b8d7a8770355993de',
        '5461/yft_labels.json': '9ef63caad8f076457d48a21986d81ddc',
        '5462/alb_labels.json': '731c74d347748b5272042f0661dad37c',
        '5463/lag_labels.json': '92d75d9218c3333ac31d74125f2b380a'
    }
    cache_subdir = os.path.abspath(os.path.join(path, 'annos'))
    url_prefix = 'https://kaggle2.blob.core.windows.net/forum-message-attachments/147157/'
    
    if not os.path.exists(cache_subdir):
        os.makedirs(cache_subdir)
    
    for url_suffix, md5_hash in annot_urls.iteritems():
        fname = url_suffix.rsplit('/', 1)[-1]
        get_file(fname, url_prefix + url_suffix, cache_subdir=cache_subdir, md5_hash=md5_hash)


# In[5]:


get_annotations()


# In[48]:


bb_json = {}
for c in anno_classes:
    if c == 'other': continue # no annotation file for "other" class
    j = json.load(open('{}annos/{}_labels.json'.format(path, c), 'r'))
    for l in j:
        if 'annotations' in l.keys() and len(l['annotations'])>0:
            bb_json[l['filename'].split('/')[-1]] = sorted(
                l['annotations'], key=lambda x: x['height']*x['width'])[-1]


# In[49]:


bb_json['img_04908.jpg']


# In[50]:


file2idx = {o:i for i,o in enumerate(raw_filenames)}
val_file2idx = {o:i for i,o in enumerate(raw_val_filenames)}


# For any images that have no annotations, we'll create an empty bounding box.

# In[51]:


empty_bbox = {'height': 0., 'width': 0., 'x': 0., 'y': 0.}


# In[52]:


for f in raw_filenames:
    if not f in bb_json.keys(): bb_json[f] = empty_bbox
for f in raw_val_filenames:
    if not f in bb_json.keys(): bb_json[f] = empty_bbox


# Finally, we convert the dictionary into an array, and convert the coordinates to our resized 224x224 images.

# In[53]:


bb_params = ['height', 'width', 'x', 'y']
def convert_bb(bb, size):
    bb = [bb[p] for p in bb_params]
    conv_x = (224. / size[0])
    conv_y = (224. / size[1])
    bb[0] = bb[0]*conv_y
    bb[1] = bb[1]*conv_x
    bb[2] = max(bb[2]*conv_x, 0)
    bb[3] = max(bb[3]*conv_y, 0)
    return bb


# In[54]:


trn_bbox = np.stack([convert_bb(bb_json[f], s) for f,s in zip(raw_filenames, sizes)], 
                   ).astype(np.float32)
val_bbox = np.stack([convert_bb(bb_json[f], s) 
                   for f,s in zip(raw_val_filenames, raw_val_sizes)]).astype(np.float32)


# Now we can check our work by drawing one of the annotations.

# In[55]:


def create_rect(bb, color='red'):
    return plt.Rectangle((bb[2], bb[3]), bb[1], bb[0], color=color, fill=False, lw=3)

def show_bb(i):
    bb = val_bbox[i]
    plot(val[i])
    plt.gca().add_patch(create_rect(bb))


# In[56]:


show_bb(0)


# ### Create & train model

# Since we're not allowed (by the kaggle rules) to manually annotate the test set, we'll need to create a model that predicts the locations of the bounding box on each image. To do so, we create a model with multiple outputs: it will predict both the type of fish (the 'class'), and the 4 bounding box coordinates. We prefer this approach to only predicting the bounding box coordinates, since we hope that giving the model more context about what it's looking for will help it with both tasks.

# In[57]:


p=0.6


# In[58]:


inp = Input(conv_layers[-1].output_shape[1:])
x = MaxPooling2D()(inp)
x = BatchNormalization(axis=1)(x)
x = Dropout(p/4)(x)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(p)(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(p/2)(x)
x_bb = Dense(4, name='bb')(x)
x_class = Dense(8, activation='softmax', name='class')(x)


# Since we have multiple outputs, we need to provide them to the model constructor in an array, and we also need to say what loss function to use for each. We also weight the bounding box loss function down by 1000x since the scale of the cross-entropy loss and the MSE is very different.

# In[59]:


model = Model([inp], [x_bb, x_class])
model.compile(Adam(lr=0.001), loss=['mse', 'categorical_crossentropy'], metrics=['accuracy'],
             loss_weights=[.001, 1.])


# In[60]:


model.fit(conv_feat, [trn_bbox, trn_labels], batch_size=batch_size, nb_epoch=3, 
             validation_data=(conv_val_feat, [val_bbox, val_labels]))


# In[61]:


model.optimizer.lr = 1e-5


# In[62]:


model.fit(conv_feat, [trn_bbox, trn_labels], batch_size=batch_size, nb_epoch=10, 
             validation_data=(conv_val_feat, [val_bbox, val_labels]))


# Excitingly, it turned out that the classification model is much improved by giving it this additional task. Let's see how well the bounding box model did by taking a look at its output.

# In[63]:


pred = model.predict(conv_val_feat[0:10])


# In[64]:


def show_bb_pred(i):
    bb = val_bbox[i]
    bb_pred = pred[0][i]
    plt.figure(figsize=(6,6))
    plot(val[i])
    ax=plt.gca()
    ax.add_patch(create_rect(bb_pred, 'yellow'))
    ax.add_patch(create_rect(bb))


# The image shows that it can find fish that are tricky for us to see!

# In[65]:


show_bb_pred(6)


# In[66]:


model.evaluate(conv_val_feat, [val_bbox, val_labels])


# In[67]:


model.save_weights(path+'models/bn_anno.h5')


# In[57]:


model.load_weights(path+'models/bn_anno.h5')


# ## Larger size

# ### Set up data

# Let's see if we get better results if we use larger images. We'll use 640x360, since it's the same shape as the most common size we saw earlier (1280x720), without being too big.

# In[68]:


trn = get_data(path+'train', (360,640))
val = get_data(path+'valid', (360,640))


# The image shows that things are much clearer at this size.

# In[70]:


plot(trn[0])


# In[71]:


test = get_data(path+'test', (360,640))


# In[72]:


save_array(path+'results/trn_640.dat', trn)
save_array(path+'results/val_640.dat', val)


# In[73]:


save_array(path+'results/test_640.dat', test)


# In[6]:


trn = load_array(path+'results/trn_640.dat')
val = load_array(path+'results/val_640.dat')


# We can now create our VGG model - we'll need to tell it we're not using the normal 224x224 images, which also means it won't include the fully connected layers (since they don't make sense for non-default sizes). We will also remove the last max pooling layer, since we don't want to throw away information yet.

# In[74]:


vgg640 = Vgg16BN((360, 640)).model
vgg640.pop()
vgg640.input_shape, vgg640.output_shape
vgg640.compile(Adam(), 'categorical_crossentropy', metrics=['accuracy'])


# We can now pre-compute the output of the convolutional part of VGG.

# In[75]:


conv_val_feat = vgg640.predict(val, batch_size=32, verbose=1)
conv_trn_feat = vgg640.predict(trn, batch_size=32, verbose=1)


# In[76]:


save_array(path+'results/conv_val_640.dat', conv_val_feat)
save_array(path+'results/conv_trn_640.dat', conv_trn_feat)


# In[77]:


conv_test_feat = vgg640.predict(test, batch_size=32, verbose=1)


# In[83]:


save_array(path+'results/conv_test_640.dat', conv_test_feat)


# In[10]:


conv_val_feat = load_array(path+'results/conv_val_640.dat')
conv_trn_feat = load_array(path+'results/conv_trn_640.dat')


# In[868]:


conv_test_feat = load_array(path+'results/conv_test_640.dat')


# ### Fully convolutional net (FCN)

# Since we're using a larger input, the output of the final convolutional layer is also larger. So we probably don't want to put a dense layer there - that would be a *lot* of parameters! Instead, let's use a fully convolutional net (FCN); this also has the benefit that they tend to generalize well, and also seems like a good fit for our problem (since the fish are a small part of the image).

# In[78]:


conv_layers,_ = split_at(vgg640, Convolution2D)


# I'm not using any dropout, since I found I got better results without it.

# In[79]:


nf=128; p=0.


# In[80]:


def get_lrg_layers():
    return [
        BatchNormalization(axis=1, input_shape=conv_layers[-1].output_shape[1:]),
        Convolution2D(nf,3,3, activation='relu', border_mode='same'),
        BatchNormalization(axis=1),
        MaxPooling2D(),
        Convolution2D(nf,3,3, activation='relu', border_mode='same'),
        BatchNormalization(axis=1),
        MaxPooling2D(),
        Convolution2D(nf,3,3, activation='relu', border_mode='same'),
        BatchNormalization(axis=1),
        MaxPooling2D((1,2)),
        Convolution2D(8,3,3, border_mode='same'),
        Dropout(p),
        GlobalAveragePooling2D(),
        Activation('softmax')
    ]


# In[81]:


lrg_model = Sequential(get_lrg_layers())


# In[ ]:


lrg_model.summary()


# In[84]:


lrg_model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


lrg_model.fit(conv_trn_feat, trn_labels, batch_size=batch_size, nb_epoch=2, 
             validation_data=(conv_val_feat, val_labels))


# In[86]:


lrg_model.optimizer.lr=1e-5


# In[365]:


lrg_model.fit(conv_trn_feat, trn_labels, batch_size=batch_size, nb_epoch=6, 
             validation_data=(conv_val_feat, val_labels))


# When I submitted the results of this model to Kaggle, I got the best single model results of any shown here (ranked 22nd on the leaderboard as at Dec-6-2016.)

# In[366]:


lrg_model.save_weights(path+'models/lrg_nmp.h5')


# In[870]:


lrg_model.load_weights(path+'models/lrg_nmp.h5')


# In[871]:


lrg_model.evaluate(conv_val_feat, val_labels)


# Another benefit of this kind of model is that the last convolutional layer has to learn to classify each part of the image (since there's only an average pooling layer after). Let's create a function that grabs the output of this layer (which is the 4th-last layer of our model).

# In[872]:


l = lrg_model.layers
conv_fn = K.function([l[0].input, K.learning_phase()], l[-4].output)


# In[881]:


def get_cm(inp, label):
    conv = conv_fn([inp,0])[0, label]
    return scipy.misc.imresize(conv, (360,640), interp='nearest')


# We have to add an extra dimension to our input since the CNN expects a 'batch' (even if it's just a batch of one).

# In[882]:


inp = np.expand_dims(conv_val_feat[0], 0)
np.round(lrg_model.predict(inp)[0],2)


# In[883]:


plt.imshow(to_plot(val[0]))


# In[885]:


cm = get_cm(inp, 0)


# The heatmap shows that (at very low resolution) the model is finding the fish!

# In[886]:


plt.imshow(cm, cmap="cool")


# ### All convolutional net heatmap

# To create a higher resolution heatmap, we'll remove all the max pooling layers, and repeat the previous steps.

# In[14]:


def get_lrg_layers():
    return [
        BatchNormalization(axis=1, input_shape=conv_layers[-1].output_shape[1:]),
        Convolution2D(nf,3,3, activation='relu', border_mode='same'),
        BatchNormalization(axis=1),
        Convolution2D(nf,3,3, activation='relu', border_mode='same'),
        BatchNormalization(axis=1),
        Convolution2D(nf,3,3, activation='relu', border_mode='same'),
        BatchNormalization(axis=1),
        Convolution2D(8,3,3, border_mode='same'),
        GlobalAveragePooling2D(),
        Activation('softmax')
    ]


# In[17]:


lrg_model = Sequential(get_lrg_layers())


# In[18]:


lrg_model.summary()


# In[19]:


lrg_model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])


# In[891]:


lrg_model.fit(conv_trn_feat, trn_labels, batch_size=batch_size, nb_epoch=2, 
             validation_data=(conv_val_feat, val_labels))


# In[892]:


lrg_model.optimizer.lr=1e-5


# In[893]:


lrg_model.fit(conv_trn_feat, trn_labels, batch_size=batch_size, nb_epoch=6, 
             validation_data=(conv_val_feat, val_labels))


# In[894]:


lrg_model.save_weights(path+'models/lrg_0mp.h5')


# In[20]:


lrg_model.load_weights(path+'models/lrg_0mp.h5')


# #### Create heatmap

# In[21]:


l = lrg_model.layers
conv_fn = K.function([l[0].input, K.learning_phase()], l[-3].output)


# In[22]:


def get_cm2(inp, label):
    conv = conv_fn([inp,0])[0, label]
    return scipy.misc.imresize(conv, (360,640))


# In[23]:


inp = np.expand_dims(conv_val_feat[0], 0)


# In[900]:


plt.imshow(to_plot(val[0]))


# In[912]:


cm = get_cm2(inp, 0)


# In[24]:


cm = get_cm2(inp, 4)


# In[913]:


plt.imshow(cm, cmap="cool")


# In[903]:


plt.figure(figsize=(10,10))
plot(val[0])
plt.imshow(cm, cmap="cool", alpha=0.5)


# ### Inception mini-net

# Here's an example of how to create and use "inception blocks" - as you see, they use multiple different convolution filter sizes and concatenate the results together. We'll talk more about these next year.

# In[198]:


def conv2d_bn(x, nb_filter, nb_row, nb_col, subsample=(1, 1)):
    x = Convolution2D(nb_filter, nb_row, nb_col,
                      subsample=subsample, activation='relu', border_mode='same')(x)
    return BatchNormalization(axis=1)(x)


# In[208]:


def incep_block(x):
    branch1x1 = conv2d_bn(x, 32, 1, 1, subsample=(2, 2))
    branch5x5 = conv2d_bn(x, 24, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 32, 5, 5, subsample=(2, 2))

    branch3x3dbl = conv2d_bn(x, 32, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 48, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 48, 3, 3, subsample=(2, 2))

    branch_pool = AveragePooling2D(
        (3, 3), strides=(2, 2), border_mode='same')(x)
    branch_pool = conv2d_bn(branch_pool, 16, 1, 1)
    return merge([branch1x1, branch5x5, branch3x3dbl, branch_pool],
              mode='concat', concat_axis=1)


# In[271]:


inp = Input(vgg640.layers[-1].output_shape[1:]) 
x = BatchNormalization(axis=1)(inp)
x = incep_block(x)
x = incep_block(x)
x = incep_block(x)
x = Dropout(0.75)(x)
x = Convolution2D(8,3,3, border_mode='same')(x)
x = GlobalAveragePooling2D()(x)
outp = Activation('softmax')(x)


# In[272]:


lrg_model = Model([inp], outp)


# In[273]:


lrg_model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])


# In[274]:


lrg_model.fit(conv_trn_feat, trn_labels, batch_size=batch_size, nb_epoch=2, 
             validation_data=(conv_val_feat, val_labels))


# In[275]:


lrg_model.optimizer.lr=1e-5


# In[277]:


lrg_model.fit(conv_trn_feat, trn_labels, batch_size=batch_size, nb_epoch=6, 
             validation_data=(conv_val_feat, val_labels))


# In[262]:


lrg_model.fit(conv_trn_feat, trn_labels, batch_size=batch_size, nb_epoch=10, 
             validation_data=(conv_val_feat, val_labels))


# In[110]:


lrg_model.save_weights(path+'models/lrg_nmp.h5')


# In[153]:


lrg_model.load_weights(path+'models/lrg_nmp.h5')


# ## Pseudo-labeling

# In[210]:


preds = model.predict([conv_test_feat, test_sizes], batch_size=batch_size*2)


# In[212]:


gen = image.ImageDataGenerator()


# In[214]:


test_batches = gen.flow(conv_test_feat, preds, batch_size=16)


# In[215]:


val_batches = gen.flow(conv_val_feat, val_labels, batch_size=4)


# In[217]:


batches = gen.flow(conv_feat, trn_labels, batch_size=44)


# In[292]:


mi = MixIterator([batches, test_batches, val_batches])


# In[220]:


bn_model.fit_generator(mi, mi.N, nb_epoch=8, validation_data=(conv_val_feat, val_labels))


# ## Submit

# In[821]:


def do_clip(arr, mx): return np.clip(arr, (1-mx)/7, mx)


# In[829]:


lrg_model.evaluate(conv_val_feat, val_labels, batch_size*2)


# In[851]:


preds = model.predict(conv_test_feat, batch_size=batch_size)


# In[852]:


preds = preds[1]


# In[25]:


test = load_array(path+'results/test_640.dat')


# In[5]:


test = load_array(path+'results/test.dat')


# In[26]:


preds = conv_model.predict(test, batch_size=32)


# In[853]:


subm = do_clip(preds,0.82)


# In[854]:


subm_name = path+'results/subm_bb.gz'


# In[855]:


# classes = sorted(batches.class_indices, key=batches.class_indices.get)
classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']


# In[856]:


submission = pd.DataFrame(subm, columns=classes)
submission.insert(0, 'image', raw_test_filenames)
submission.head()


# In[857]:


submission.to_csv(subm_name, index=False, compression='gzip')


# In[858]:


FileLink(subm_name)

