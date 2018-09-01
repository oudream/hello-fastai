
# coding: utf-8

# In[1]:


from theano.sandbox import cuda
cuda.use('gpu2')


# In[2]:


get_ipython().magic(u'matplotlib inline')
import utils; reload(utils)
from utils import *
from __future__ import division, print_function


# In[144]:


from keras.layers import TimeDistributed, Activation
from numpy.random import choice


# ## Setup

# We haven't really looked into the detail of how this works yet - so this is provided for self-study for those who are interested. We'll look at it closely next week.

# In[107]:


path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
text = open(path).read().lower()
print('corpus length:', len(text))


# In[272]:


get_ipython().system(u'tail {path} -n25')


# In[101]:


#path = 'data/wiki/'
#text = open(path+'small.txt').read().lower()
#print('corpus length:', len(text))

#text = text[0:1000000]


# In[124]:


chars = sorted(list(set(text)))
vocab_size = len(chars)+1
print('total chars:', vocab_size)


# In[134]:


chars.insert(0, "\0")


# In[270]:


''.join(chars[1:-6])


# In[135]:


char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


# In[136]:


idx = [char_indices[c] for c in text]


# In[276]:


idx[:10]


# In[274]:


''.join(indices_char[i] for i in idx[:70])


# ## Preprocess and create model

# In[139]:


maxlen = 40
sentences = []
next_chars = []
for i in range(0, len(idx) - maxlen+1):
    sentences.append(idx[i: i + maxlen])
    next_chars.append(idx[i+1: i+maxlen+1])
print('nb sequences:', len(sentences))


# In[ ]:


sentences = np.concatenate([[np.array(o)] for o in sentences[:-2]])
next_chars = np.concatenate([[np.array(o)] for o in next_chars[:-2]])


# In[277]:


sentences.shape, next_chars.shape


# In[213]:


n_fac = 24


# In[232]:


model=Sequential([
        Embedding(vocab_size, n_fac, input_length=maxlen),
        LSTM(512, input_dim=n_fac,return_sequences=True, dropout_U=0.2, dropout_W=0.2,
             consume_less='gpu'),
        Dropout(0.2),
        LSTM(512, return_sequences=True, dropout_U=0.2, dropout_W=0.2,
             consume_less='gpu'),
        Dropout(0.2),
        TimeDistributed(Dense(vocab_size)),
        Activation('softmax')
    ])    


# In[233]:


model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam())


# ## Train

# In[219]:


def print_example():
    seed_string="ethics is a basic foundation of all that"
    for i in range(320):
        x=np.array([char_indices[c] for c in seed_string[-40:]])[np.newaxis,:]
        preds = model.predict(x, verbose=0)[0][-1]
        preds = preds/np.sum(preds)
        next_char = choice(chars, p=preds)
        seed_string = seed_string + next_char
    print(seed_string)


# In[236]:


model.fit(sentences, np.expand_dims(next_chars,-1), batch_size=64, nb_epoch=1)


# In[220]:


print_example()


# In[236]:


model.fit(sentences, np.expand_dims(next_chars,-1), batch_size=64, nb_epoch=1)


# In[222]:


print_example()


# In[235]:


model.optimizer.lr=0.001


# In[236]:


model.fit(sentences, np.expand_dims(next_chars,-1), batch_size=64, nb_epoch=1)


# In[237]:


print_example()


# In[250]:


model.optimizer.lr=0.0001


# In[239]:


model.fit(sentences, np.expand_dims(next_chars,-1), batch_size=64, nb_epoch=1)


# In[240]:


print_example()


# In[242]:


model.save_weights('data/char_rnn.h5')


# In[257]:


model.optimizer.lr=0.00001


# In[243]:


model.fit(sentences, np.expand_dims(next_chars,-1), batch_size=64, nb_epoch=1)


# In[249]:


print_example()


# In[258]:


model.fit(sentences, np.expand_dims(next_chars,-1), batch_size=64, nb_epoch=1)


# In[264]:


print_example()


# In[283]:


print_example()


# In[282]:


model.save_weights('data/char_rnn.h5')

