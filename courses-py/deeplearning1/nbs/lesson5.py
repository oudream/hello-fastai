
# coding: utf-8

# In[1]:


from theano.sandbox import cuda


# In[2]:


get_ipython().magic(u'matplotlib inline')
import utils; reload(utils)
from utils import *
from __future__ import division, print_function


# In[3]:


model_path = 'data/imdb/models/'
get_ipython().magic(u'mkdir -p $model_path')


# ## Setup data

# We're going to look at the IMDB dataset, which contains movie reviews from IMDB, along with their sentiment. Keras comes with some helpers for this dataset.

# In[4]:


from keras.datasets import imdb
idx = imdb.get_word_index()


# This is the word list:

# In[5]:


idx_arr = sorted(idx, key=idx.get)
idx_arr[:10]


# ...and this is the mapping from id to word

# In[6]:


idx2word = {v: k for k, v in idx.iteritems()}


# We download the reviews using code copied from keras.datasets:

# In[ ]:


path = get_file('imdb_full.pkl',
                origin='https://s3.amazonaws.com/text-datasets/imdb_full.pkl',
                md5_hash='d091312047c43cf9e4e38fef92437263')
f = open(path, 'rb')
(x_train, labels_train), (x_test, labels_test) = pickle.load(f)


# In[ ]:


len(x_train)


# Here's the 1st review. As you see, the words have been replaced by ids. The ids can be looked up in idx2word.

# In[ ]:


', '.join(map(str, x_train[0]))


# The first word of the first review is 23022. Let's see what that is.

# In[ ]:


idx2word[23022]


# Here's the whole review, mapped from ids to words.

# In[ ]:


' '.join([idx2word[o] for o in x_train[0]])


# The labels are 1 for positive, 0 for negative.

# In[26]:


labels_train[:10]


# Reduce vocab size by setting rare words to max index.

# In[27]:


vocab_size = 5000

trn = [np.array([i if i<vocab_size-1 else vocab_size-1 for i in s]) for s in x_train]
test = [np.array([i if i<vocab_size-1 else vocab_size-1 for i in s]) for s in x_test]


# Look at distribution of lengths of sentences.

# In[29]:


lens = np.array(map(len, trn))
(lens.max(), lens.min(), lens.mean())


# Pad (with zero) or truncate each sentence to make consistent length.

# In[30]:


seq_len = 500

trn = sequence.pad_sequences(trn, maxlen=seq_len, value=0)
test = sequence.pad_sequences(test, maxlen=seq_len, value=0)


# This results in nice rectangular matrices that can be passed to ML algorithms. Reviews shorter than 500 words are pre-padded with zeros, those greater are truncated.

# In[32]:


trn.shape


# ## Create simple models

# ### Single hidden layer NN

# The simplest model that tends to give reasonable results is a single hidden layer net. So let's try that. Note that we can't expect to get any useful results by feeding word ids directly into a neural net - so instead we use an embedding to replace them with a vector of 32 (initially random) floats for each word in the vocab.

# In[35]:


model = Sequential([
    Embedding(vocab_size, 32, input_length=seq_len),
    Flatten(),
    Dense(100, activation='relu'),
    Dropout(0.7),
    Dense(1, activation='sigmoid')])


# In[36]:


model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
model.summary()


# In[19]:


model.fit(trn, labels_train, validation_data=(test, labels_test), nb_epoch=2, batch_size=64)


# The [stanford paper](http://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf) that this dataset is from cites a state of the art accuracy (without unlabelled data) of 0.883. So we're short of that, but on the right track.

# ### Single conv layer with max pooling

# A CNN is likely to work better, since it's designed to take advantage of ordered data. We'll need to use a 1D CNN, since a sequence of words is 1D.

# In[37]:


conv1 = Sequential([
    Embedding(vocab_size, 32, input_length=seq_len, dropout=0.2),
    Dropout(0.2),
    Convolution1D(64, 5, border_mode='same', activation='relu'),
    Dropout(0.2),
    MaxPooling1D(),
    Flatten(),
    Dense(100, activation='relu'),
    Dropout(0.7),
    Dense(1, activation='sigmoid')])


# In[45]:


conv1.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])


# In[278]:


conv1.fit(trn, labels_train, validation_data=(test, labels_test), nb_epoch=4, batch_size=64)


# That's well past the Stanford paper's accuracy - another win for CNNs!

# In[281]:


conv1.save_weights(model_path + 'conv1.h5')


# In[46]:


conv1.load_weights(model_path + 'conv1.h5')


# ## Pre-trained vectors

# You may want to look at wordvectors.ipynb before moving on.
# 
# In this section, we replicate the previous CNN, but using pre-trained embeddings.

# In[1]:


def get_glove_dataset(dataset):
    """Download the requested glove dataset from files.fast.ai
    and return a location that can be passed to load_vectors.
    """
    # see wordvectors.ipynb for info on how these files were
    # generated from the original glove data.
    md5sums = {'6B.50d': '8e1557d1228decbda7db6dfd81cd9909',
               '6B.100d': 'c92dbbeacde2b0384a43014885a60b2c',
               '6B.200d': 'af271b46c04b0b2e41a84d8cd806178d',
               '6B.300d': '30290210376887dcc6d0a5a6374d8255'}
    glove_path = os.path.abspath('data/glove/results')
    get_ipython().magic(u'mkdir -p $glove_path')
    return get_file(dataset,
                    'http://files.fast.ai/models/glove/' + dataset + '.tgz',
                    cache_subdir=glove_path,
                    md5_hash=md5sums.get(dataset, None),
                    untar=True)


# In[2]:


def load_vectors(loc):
    return (load_array(loc+'.dat'),
        pickle.load(open(loc+'_words.pkl','rb')),
        pickle.load(open(loc+'_idx.pkl','rb')))


# In[3]:


vecs, words, wordidx = load_vectors(get_glove_dataset('6B.50d'))


# The glove word ids and imdb word ids use different indexes. So we create a simple function that creates an embedding matrix using the indexes from imdb, and the embeddings from glove (where they exist).

# In[73]:


def create_emb():
    n_fact = vecs.shape[1]
    emb = np.zeros((vocab_size, n_fact))

    for i in range(1,len(emb)):
        word = idx2word[i]
        if word and re.match(r"^[a-zA-Z0-9\-]*$", word):
            src_idx = wordidx[word]
            emb[i] = vecs[src_idx]
        else:
            # If we can't find the word in glove, randomly initialize
            emb[i] = normal(scale=0.6, size=(n_fact,))

    # This is our "rare word" id - we want to randomly initialize
    emb[-1] = normal(scale=0.6, size=(n_fact,))
    emb/=3
    return emb


# In[21]:


emb = create_emb()


# We pass our embedding matrix to the Embedding constructor, and set it to non-trainable.

# In[87]:


model = Sequential([
    Embedding(vocab_size, 50, input_length=seq_len, dropout=0.2, 
              weights=[emb], trainable=False),
    Dropout(0.25),
    Convolution1D(64, 5, border_mode='same', activation='relu'),
    Dropout(0.25),
    MaxPooling1D(),
    Flatten(),
    Dense(100, activation='relu'),
    Dropout(0.7),
    Dense(1, activation='sigmoid')])


# In[88]:


model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])


# In[90]:


model.fit(trn, labels_train, validation_data=(test, labels_test), nb_epoch=2, batch_size=64)


# We already have beaten our previous model! But let's fine-tune the embedding weights - especially since the words we couldn't find in glove just have random embeddings.

# In[91]:


model.layers[0].trainable=True


# In[92]:


model.optimizer.lr=1e-4


# In[93]:


model.fit(trn, labels_train, validation_data=(test, labels_test), nb_epoch=1, batch_size=64)


# As expected, that's given us a nice little boost. :)

# In[94]:


model.save_weights(model_path+'glove50.h5')


# ## Multi-size CNN

# This is an implementation of a multi-size CNN as shown in Ben Bowles' [excellent blog post](https://quid.com/feed/how-quid-uses-deep-learning-with-small-data).

# In[23]:


from keras.layers import Merge


# We use the functional API to create multiple conv layers of different sizes, and then concatenate them.

# In[132]:


graph_in = Input ((vocab_size, 50))
convs = [ ] 
for fsz in range (3, 6): 
    x = Convolution1D(64, fsz, border_mode='same', activation="relu")(graph_in)
    x = MaxPooling1D()(x) 
    x = Flatten()(x) 
    convs.append(x)
out = Merge(mode="concat")(convs) 
graph = Model(graph_in, out) 


# In[174]:


emb = create_emb()


# We then replace the conv/max-pool layer in our original CNN with the concatenated conv layers.

# In[175]:


model = Sequential ([
    Embedding(vocab_size, 50, input_length=seq_len, dropout=0.2, weights=[emb]),
    Dropout (0.2),
    graph,
    Dropout (0.5),
    Dense (100, activation="relu"),
    Dropout (0.7),
    Dense (1, activation='sigmoid')
    ])


# In[176]:


model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])


# In[177]:


model.fit(trn, labels_train, validation_data=(test, labels_test), nb_epoch=2, batch_size=64)


# Interestingly, I found that in this case I got best results when I started the embedding layer as being trainable, and then set it to non-trainable after a couple of epochs. I have no idea why!

# In[178]:


model.layers[0].trainable=False


# In[179]:


model.optimizer.lr=1e-5


# In[180]:


model.fit(trn, labels_train, validation_data=(test, labels_test), nb_epoch=2, batch_size=64)


# This more complex architecture has given us another boost in accuracy.

# ## LSTM

# We haven't covered this bit yet!

# In[79]:


model = Sequential([
    Embedding(vocab_size, 32, input_length=seq_len, mask_zero=True,
              W_regularizer=l2(1e-6), dropout=0.2),
    LSTM(100, consume_less='gpu'),
    Dense(1, activation='sigmoid')])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


# In[80]:


model.fit(trn, labels_train, validation_data=(test, labels_test), nb_epoch=5, batch_size=64)

