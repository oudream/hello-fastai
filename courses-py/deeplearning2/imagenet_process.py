
# coding: utf-8

# # Imagenet Processing in parallel

# In[3]:


get_ipython().magic(u'matplotlib inline')
import importlib
import utils2; importlib.reload(utils2)
from utils2 import *


# In[4]:


from bcolz_array_iterator import BcolzArrayIterator
from tqdm import tqdm


# In[5]:


limit_mem()


# This is where our full dataset lives. It's slow spinning discs, but lots of room!
# 
# **NB:** We can easily switch to and from using a sample. We'll use a sample for *everything*, except the final complete processing (which we'll use fast/expensive compute for, and time on the sample so we know how long it will take).

# In[6]:


path = '/data/jhoward/imagenet/full/'
# path = '/data/jhoward/imagenet/sample/'


# This is on a RAID 1 SSD for fast access, so good for resized images and feature arrays

# In[7]:


dpath = '/data/jhoward/fast/imagenet/full/'
# dpath = '/data/jhoward/fast/imagenet/sample/'
# %mkdir {dpath}


# Note that either way, AWS isn't going to be a great place for doing this kind of analysis - putting a model into production will cost at minimum $600/month for a P2 instance. For that price you can **buy** a GTX 1080 card, which has **double** the performance of the AWS P2 card! And you can set up your slow full data RAID 5 array and your fast preprocessed data RAID 1 array just as you like it. Since you'll want your own servers for production, you may as well use them for training, and benefit from the greater speed, lower cost, and greater control of storage resources.
# 
# You can put your server inside a colo facility for very little money, paying just for the network and power. (Cloud providers aren't even allowed to provide GTX 1080's!)
# 
# There's little need for distributed computing systems for the vast majority of training and production needs in deep learning.

# ## Get word vectors

# First we need to grab some word vectors, to use as our dependent variable for the image model (so that the image vectors and word vectors will be in the same space). After loading the word vectors, we'll make sure that the names of the wordnet/imagenet are in the word list.

# * Be careful not to just follow paper's approach - e.g. here word2vec better than custom wikipedia vectors. word2vec has multi-word tokens like 'golden retriever'
# * Take evaluations shown in papers with a grain of salt, and do your own tests on important bits. E.g. DeVISE (because it's an older paper) used an old and inaccurate image model, and poor word vectors, so recent papers that compare to it aren't so relevent

# In[83]:


from gensim.models import word2vec
w2v_path='/data/jhoward/datasets/nlp/GoogleNews-vectors-negative300'


# In[87]:


model = word2vec.KeyedVectors.load_word2vec_format(w2v_path+'.bin', binary=True)
model.save_word2vec_format(w2v_path+'.txt', binary=False)


# In[88]:


lines = open(w2v_path+'.txt').readlines()


# In[89]:


def parse_w2v(l):
    i=l.index(' ')
    return l[:i], np.fromstring(l[i+1:-2], 'float32', sep=' ')


# In[90]:


w2v_list = list(map(parse_w2v, lines[1:]))


# In[91]:


pickle.dump(w2v_list, open(path+'../w2vl.pkl', 'wb'))


# In[8]:


w2v_list = pickle.load(open(path+'../w2vl.pkl', 'rb'))


# We save the processed file so we can access it quickly in the future. It's a good idea to save any intermediate results that take a while to recreate, so you can use them both in production and prototyping.

# In[9]:


w2v_dict = dict(w2v_list)
words,vectors = zip(*w2v_list)


# Always test your inputs! If you're not sure what to look for, try to come up with some kind of reasonableness test.

# In[10]:


np.corrcoef(w2v_dict['jeremy'], w2v_dict['Jeremy'])


# In[11]:


np.corrcoef(w2v_dict['banana'], w2v_dict['Jeremy'])


# In[12]:


lc_w2v = {w.lower(): w2v_dict[w] for w in reversed(words)}


# We're going to map word vectors for each of:
# * The 1000 categories in the Imagenet competition
# * The 82,000 nouns in Wordnet

# In[13]:


fpath = get_file('imagenet_class_index.json', 
                 'http://www.platform.ai/models/imagenet_class_index.json', 
                 cache_subdir='models')
class_dict = json.load(open(fpath))
nclass = len(class_dict); nclass


# In[14]:


classids_1k = dict(class_dict.values())
classid_lines = open(path+'../classids.txt', 'r').readlines()
classids = dict(l.strip().split(' ') for l in classid_lines)
len(classids)


# In[15]:


syn_wv = [(k, lc_w2v[v.lower()]) for k,v in classids.items()
          if v.lower() in lc_w2v]
syn_wv_1k = [(k, lc_w2v[v.lower()]) for k,v in classids_1k.items()
          if v.lower() in lc_w2v]
syn2wv = dict(syn_wv); len(syn2wv)


# In[16]:


nomatch = [v[0] for v in class_dict.values() if v[0] not in syn2wv]


# In[17]:


# nm_path=path+'train_nm/'
# os.mkdir(nm_path)
# for nm in nomatch: os.rename(path+'train/'+nm, nm_path+nm)


# In[18]:


ndim = len(list(syn2wv.values())[0]); ndim


# ## Resize images

# Now that we've got our word vectors, we need a model that can create image vectors. It's nearly always best to start with a pre-train image model, and these require a specific size input. We'll be using resnet, which requires 224x224 sized images. Reading jpegs and resizing them can be slow, so we'll store the result of this.
# 
# First we create the filename list for the imagenet archive:

# In[102]:


fnames = list(glob.iglob(path+'train/*/*.JPEG'))
pickle.dump(fnames, open(path+'fnames.pkl', 'wb'))


# Even scanning a large collection of files is slow, so we save the filenames:

# In[18]:


fnames = pickle.load(open(path+'fnames.pkl', 'rb'))


# In[19]:


fnames = np.random.permutation(fnames)


# In[21]:


pickle.dump(fnames, open(path+'fnames_r.pkl', 'wb'))


# In[19]:


fnames = pickle.load(open(path+'fnames_r.pkl', 'rb'))


# In[20]:


new_s = 224 # height and width to resize to
n = len(fnames); n


# In[21]:


bc_path = f'{dpath}/trn_resized_{new_s}_r.bc'


# In[22]:


bc_path = f'{path}/results/trn_resized_{new_s}_r.bc'


# Using pillow to resize the image (recommendation: install pillow-simd for 600% speedup). To install, force remove the conda installed version, then:
# 
#     CC="cc -mavx2" pip install -U --force-reinstall pillow-simd

# In[139]:


def _resize(img):
    shortest = min(img.width,img.height)
    resized = np.round(np.multiply(new_s/shortest, img.size)).astype(int)
    return img.resize(resized, Image.BILINEAR)


# In[140]:


def resize_img(i):
    img = Image.open(fnames[i])
    s = np.array(img).shape
    if len(s)!=3 or s[2]!=3: return
    return _resize(img)


# In[141]:


def resize_img_bw(i):
    return _resize(Image.open(fnames[i]).convert('L'))


# Pre-allocate memory in threadlocal storage

# In[142]:


tl = threading.local()


# In[143]:


tl.place = np.zeros((new_s,new_s,3), 'uint8')
#tl.place = np.zeros((new_s,new_s), 'uint8')


# Bcolz is amazingly fast, easy to use, and provides a largely numpy-compatible interface. It creates file-backed arrays and are transparently cached in memory.
# 
# Create (or open) compressed array for our resized images

# In[150]:


arr = bcolz.carray(np.empty((0, new_s, new_s, 3), 'float32'), 
                   chunklen=16, mode='w', rootdir=bc_path)


# Function that appends resized image with black border added to longer axis

# In[145]:


def get_slice(p, n): return slice((p-n+1)//2, p-(p-n)//2)

def app_img(r):
    tl.place[:] = (np.array(r)[get_slice(r.size[1],new_s), get_slice(r.size[0],new_s)] 
        if r else 0.)
    arr.append(tl.place)


# In[241]:


# Serial version
for i in range(2000): app_img(resize_img(i))
arr.flush()


# In[151]:


# Parallel version
step=6400
for i in tqdm(range(0, n, step)):
    with ThreadPoolExecutor(max_workers=16) as execr:
        res = execr.map(resize_img, range(i, min(i+step, n)))
        for r in res: app_img(r)
    arr.flush()


# Times to process 2000 images that aren't in filesystem cache (tpe==ThreadPoolExecutor, ppe==ProcessPoolExecutor; number shows #jobs)

# In[115]:


times = [('tpe 16', 3.22), ('tpe 12', 3.65), ('ppe 12', 3.97), ('ppe 8 ', 4.47), 
         ('ppe 6 ', 4.89), ('ppe 3 ', 8.03), ('serial', 25.3)]

column_chart(*zip(*times))


# In[23]:


arr = bcolz.open(bc_path)


# In[24]:


plt.imshow(arr[-2].astype('uint8'))


# We do our prototyping in a notebook, and then use *'Download as->Notebook'* to get a python script we can run under tmux. Notebooks are great for running small experiments, since it's easy to make lots of changes and inspect the results in a wide variety of ways.

# ## Create model

# Now we're ready to create our first model. Step one: create our target labels, which is simply a case of grabbing the synset id from the filename, and looking up the word vector for each.

# In[25]:


def get_synset(f): return f[f.rfind('/')+1:f.find('_')]

labels = list(map(get_synset, fnames))
labels[:5]


# In[26]:


vecs = np.stack([syn2wv[l] for l in labels]); vecs.shape


# We'll be using resnet as our model for these experiments.

# In[27]:


rn_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((1,1,3))
inp = Input((224,224,3))
preproc = Lambda(lambda x: (x - rn_mean)[:, :, :, ::-1])(inp)
model = ResNet50(include_top=False, input_tensor=preproc)


# In order to make each step faster, we'll save a couple of intermediate activations that we'll be using shortly. First, the last layer before the final convolutional bottleneck:

# In[28]:


mid_start = model.get_layer('res5b_branch2a')
mid_out = model.layers[model.layers.index(mid_start)-1]
shp=mid_out.output_shape; shp


# We put an average pooling layer on top to make it a more managable size.

# In[29]:


rn_top = Model(model.input, mid_out.output)
rn_top_avg = Sequential([rn_top, AveragePooling2D((7,7))])


# In[30]:


shp=rn_top_avg.output_shape; shp


# We create this intermediate array a batch at a time, so we don't have to keep it in memory.

# In[31]:


features_mid = bcolz.open(path+'results/features_mid_1c_r.bc')


# In[30]:


features_mid = bcolz.carray(np.empty((0,)+shp[1:]), rootdir=path+'results/features_mid_1c_r.bc',
                           chunklen=16, mode='w')


# In[47]:


def gen_features_mid(dirn):
    gen = (features_mid[i:min(i+128,n)] for i in range(0, len(features_mid), 128))
    for i,batch in tqdm(enumerate(gen)):
        features_mid2.append(rn_top_avg.predict(batch[:,:,::dirn]))
        if (i%100==99): features_mid2.flush()
    features_mid2.flush()


# In[52]:


gen_features_mid(1)


# In[34]:


gen_features_mid(-1)


# In[54]:


features_mid.shape


# Our final layers match the original resnet, although we add on extra resnet block at the top as well.

# In[46]:


rn_bot_inp = Input(shp[1:])
x = rn_bot_inp
# x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
# x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
x = Flatten()(x)
rn_bot = Model(rn_bot_inp, x)
rn_bot.output_shape


# In[47]:


for i in range(len(rn_bot.layers)-2):
    rn_bot.layers[-i-2].set_weights(model.layers[-i-2].get_weights())


# We save this layer's results too, although it's smaller so should fit in RAM.

# In[59]:


get_ipython().magic(u'time features_last = rn_bot.predict(features_mid, batch_size=128)')


# In[60]:


features_last = bcolz.carray(features_last, rootdir=path+'results/features_last_r.bc', 
                             chunklen=64, mode='w')


# In[32]:


features_last = bcolz.open(path+'results/features_last_r.bc')[:]


# We add a linear model on top to predict our word vectors.

# In[33]:


lm_inp = Input(shape=(2048,))
lm = Model(lm_inp, Dense(ndim)(lm_inp))


# cosine distance is a good choice for anything involving nearest neighbors (which we'll use later).

# In[34]:


def cos_distance(y_true, y_pred):
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return K.mean(1 - K.sum((y_true * y_pred), axis=-1))


# In[35]:


lm.compile('adam','cosine_proximity')


# In[68]:


lm.evaluate(features_last, vecs, verbose=0)


# In[69]:


lm.fit(features_last, v, verbose=2, nb_epoch=3)


# Be sure to save intermediate weights, to avoid recalculating them

# In[70]:


lm.save_weights(path+'results/lm_cos.h5')


# In[36]:


lm.load_weights(path+'results/lm_cos.h5')


# ## Nearest Neighbors

# Let's use nearest neighbors to look at a couple of examples, to see how well it's working. The first NN will be just looking at the word vectors of the 1,000 imagenet competition categories.

# In[37]:


syns, wvs = list(zip(*syn_wv_1k))
wvs = np.array(wvs)


# In[38]:


nn = NearestNeighbors(3, metric='cosine', algorithm='brute').fit(wvs)


# In[39]:


nn = LSHForest(20, n_neighbors=3).fit(wvs)


# In[40]:


get_ipython().magic(u'time pred_wv = lm.predict(features_last[:10000])')


# In[41]:


get_ipython().magic(u'time dist, idxs = nn.kneighbors(pred_wv)')


# In[42]:


[[classids[syns[id]] for id in ids] for ids in idxs[190:200]]


# In[77]:


plt.imshow(arr[190].astype('uint8'))


# A much harder task is to look up every wordnet synset id.

# In[78]:


all_syns, all_wvs = list(zip(*syn_wv))
all_wvs = np.array(all_wvs)


# In[79]:


all_nn = LSHForest(20, n_neighbors=3).fit(all_wvs)


# In[80]:


get_ipython().magic(u'time dist, idxs = all_nn.kneighbors(pred_wv[:200])')


# In[81]:


[[classids[all_syns[id]] for id in ids] for ids in idxs[190:200]]


# ## Fine tune

# To improve things, let's fine tune more layers.

# In[43]:


lm_inp2 = Input(shape=(2048,))
lm2 = Model(lm_inp2, Dense(ndim)(lm_inp2))


# In[44]:


for l1,l2 in zip(lm.layers,lm2.layers): l2.set_weights(l1.get_weights())


# In[48]:


rn_bot_seq = Sequential([rn_bot, lm2])
rn_bot_seq.compile('adam', 'cosine_proximity')
rn_bot_seq.output_shape


# In[85]:


bc_it = BcolzArrayIterator(features_mid, v, shuffle=True, batch_size=128)


# In[86]:


K.set_value(rn_bot_seq.optimizer.lr, 1e-3)


# In[87]:


rn_bot_seq.fit_generator(bc_it, bc_it.N, verbose=2, nb_epoch=4)


# In[88]:


K.set_value(rn_bot_seq.optimizer.lr, 1e-4)


# In[89]:


rn_bot_seq.fit_generator(bc_it, bc_it.N, verbose=2, nb_epoch=8)


# In[90]:


K.set_value(rn_bot_seq.optimizer.lr, 1e-5)


# In[91]:


rn_bot_seq.fit_generator(bc_it, bc_it.N, verbose=2, nb_epoch=5)


# In[92]:


rn_bot_seq.evaluate(features_mid, v, verbose=2)


# In[93]:


rn_bot_seq.save_weights(path+'results/rn_bot_seq_cos.h5')


# In[49]:


rn_bot_seq.load_weights(path+'results/rn_bot_seq_cos.h5')


# ## KNN again

# In[54]:


get_ipython().magic(u'time pred_wv = rn_bot_seq.predict(features_mid)')


# In[55]:


rng = slice(190,200)


# In[56]:


dist, idxs = nn.kneighbors(pred_wv[rng])


# In[57]:


[[classids[syns[id]] for id in ids] for ids in idxs]


# In[98]:


dist, idxs = all_nn.kneighbors(pred_wv[rng])


# In[99]:


[[classids[all_syns[id]] for id in ids] for ids in idxs]


# In[100]:


plt.imshow(arr[rng][1].astype('uint8'))


# ## Text -> Image

# Something very nice about this kind of model is we can go in the other direction as well - find images similar to a word or phrase!

# In[58]:


img_nn = NearestNeighbors(3, metric='cosine', algorithm='brute').fit(pred_wv)


# In[60]:


img_nn2 = LSHForest(20, n_neighbors=3).fit(pred_wv)


# In[116]:


word = 'boat'
vec = w2v_dict[word]
# dist, idxs = img_nn.kneighbors(vec.reshape(1,-1))
dist, idxs = img_nn2.kneighbors(vec.reshape(1,-1))


# In[133]:


ims = [Image.open(fnames[fn%n]) for fn in idxs[0]]
display(*ims)


# In[132]:


vec = (w2v_dict['engine'] + w2v_dict['boat'])/2
dist, idxs = img_nn.kneighbors(vec.reshape(1,-1))


# In[82]:


def slerp(val, low, high):
    """Spherical interpolation. val has a range of 0 to 1."""
    if val <= 0: return low
    elif val >= 1: return high
    omega = np.arccos(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)))
    so = np.sin(omega)
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega)/so * high


# In[128]:


vec = slerp(0.5, w2v_dict['paddle'], w2v_dict['boat'])
dist, idxs = img_nn.kneighbors(vec.reshape(1,-1))


# ## Image -> image

# Since that worked so well, let's try to find images with similar content to another image...

# In[105]:


ft_model = Sequential([rn_top_avg, rn_bot_seq])


# In[106]:


new_file = '/data/jhoward/imagenet/full/valid/n01498041/ILSVRC2012_val_00005642.JPEG'


# In[107]:


new_file = '/data/jhoward/imagenet/full/valid/n01440764/ILSVRC2012_val_00007197.JPEG'


# In[108]:


new_im = Image.open(new_file).resize((224,224), Image.BILINEAR); new_im


# In[109]:


vec = ft_model.predict(np.expand_dims(new_im, 0))


# In[110]:


dist, idxs = img_nn2.kneighbors(vec)


# In[111]:


ims = [Image.open(fnames[fn%n]) for fn in idxs[0]]
display(*ims)

