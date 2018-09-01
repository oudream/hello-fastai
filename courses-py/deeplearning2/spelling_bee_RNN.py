
# coding: utf-8

# # Spelling Bee

# This notebook starts our deep dive (no pun intended) into NLP by introducing sequence-to-sequence learning on Spelling Bee.

# ## Data Stuff
# 
# We take our data set from [The CMU pronouncing dictionary](https://en.wikipedia.org/wiki/CMU_Pronouncing_Dictionary)

# In[1]:


get_ipython().magic(u'matplotlib inline')
import importlib
import utils2; importlib.reload(utils2)
from utils2 import *
np.set_printoptions(4)
PATH = 'data/spellbee/'


# In[2]:


limit_mem()


# In[3]:


from sklearn.model_selection import train_test_split


# The CMU pronouncing dictionary consists of sounds/words and their corresponding phonetic description (American pronunciation).
# 
# The phonetic descriptions are a sequence of phonemes. Note that the vowels end with integers; these indicate where the stress is.
# 
# Our goal is to learn how to spell these words given the sequence of phonemes.

# The preparation of this data set follows the same pattern we've seen before for NLP tasks.
# 
# Here we iterate through each line of the file and grab each word/phoneme pair that starts with an uppercase letter. 

# In[4]:


lines = [l.strip().split("  ") for l in open(PATH+"cmudict-0.7b", encoding='latin1') 
         if re.match('^[A-Z]', l)]
lines = [(w, ps.split()) for w, ps in lines]
lines[0], lines[-1]


# Next we're going to get a list of the unique phonemes in our vocabulary, as well as add a null "_" for zero-padding.

# In[5]:


phonemes = ["_"] + sorted(set(p for w, ps in lines for p in ps))
phonemes[:5]


# In[6]:


len(phonemes)


# Then we create mappings of phonemes and letters to respective indices.
# 
# Our letters include the padding element "_", but also "*" which we'll explain later.

# In[7]:


p2i = dict((v, k) for k,v in enumerate(phonemes))
letters = "_abcdefghijklmnopqrstuvwxyz*"
l2i = dict((v, k) for k,v in enumerate(letters))


# Let's create a dictionary mapping words to the sequence of indices corresponding to it's phonemes, and let's do it only for words between 5 and 15 characters long.

# In[8]:


maxlen=15
pronounce_dict = {w.lower(): [p2i[p] for p in ps] for w, ps in lines
                 if (5<=len(w)<=maxlen) and re.match("^[A-Z]+$", w)}
len(pronounce_dict)


# Aside on various approaches to python's list comprehension:
# * the first list is a typical example of a list comprehension subject to a conditional
# * the second is a list comprehension inside a list comprehension, which returns a list of list
# * the third is similar to the second, but is read and behaves like a nested loop
#     * Since there is no inner bracket, there are no lists wrapping the inner loop

# In[9]:


a=['xyz','abc']
[o.upper() for o in a if o[0]=='x'], [[p for p in o] for o in a], [p for o in a for p in o]


# Split lines into words, phonemes, convert to indexes (with padding), split into training, validation, test sets. Note we also find the max phoneme sequence length for padding.

# In[10]:


maxlen_p = max([len(v) for k,v in pronounce_dict.items()])


# In[11]:


pairs = np.random.permutation(list(pronounce_dict.keys()))
n = len(pairs)
input_ = np.zeros((n, maxlen_p), np.int32)
labels_ = np.zeros((n, maxlen), np.int32)

for i, k in enumerate(pairs):
    for j, p in enumerate(pronounce_dict[k]): input_[i][j] = p
    for j, letter in enumerate(k): labels_[i][j] = l2i[letter]


# In[12]:


go_token = l2i["*"]
dec_input_ = np.concatenate([np.ones((n,1)) * go_token, labels_[:,:-1]], axis=1)


# Sklearn's <tt>train_test_split</tt> is an easy way to split data into training and testing sets.

# In[13]:


(input_train, input_test, labels_train, labels_test, dec_input_train, dec_input_test
    ) = train_test_split(input_, labels_, dec_input_, test_size=0.1)


# In[14]:


input_train.shape


# In[15]:


labels_train.shape


# In[16]:


input_vocab_size, output_vocab_size = len(phonemes), len(letters)
input_vocab_size, output_vocab_size


# Next we proceed to build our model.

# ## Keras code

# In[17]:


parms = {'verbose': 0, 'callbacks': [TQDMNotebookCallback(leave_inner=True)]}
lstm_params = {}


# In[18]:


dim = 240


# ### Without attention

# In[19]:


def get_rnn(return_sequences= True): 
    return LSTM(dim, dropout_U= 0.1, dropout_W= 0.1, 
               consume_less= 'gpu', return_sequences=return_sequences)


# The model has three parts:
# * We first pass list of phonemes through an embedding function to get a list of phoneme embeddings. Our goal is to turn this sequence of embeddings into a single distributed representation that captures what our phonemes say.
# * Turning a sequence into a representation can be done using an RNN. This approach is useful because RNN's are able to keep track of state and memory, which is obviously important in forming a complete understanding of a pronunciation.
#     * <tt>BiDirectional</tt> passes the original sequence through an RNN, and the reversed sequence through a different RNN and concatenates the results. This allows us to look forward and backwards.
#     * We do this because in language things that happen later often influence what came before (i.e. in Spanish, "el chico, la chica" means the boy, the girl; the word for "the" is determined by the gender of the subject, which comes after).
# * Finally, we arrive at a vector representation of the sequence which captures everything we need to spell it. We feed this vector into more RNN's, which are trying to generate the labels. After this, we make a classification for what each letter is in the output sequence.
#     * We use <tt>RepeatVector</tt> to help our RNN remember at each point what the original word is that it's trying to translate.
#     
# 

# In[20]:


inp = Input((maxlen_p,))
x = Embedding(input_vocab_size, 120)(inp)

x = Bidirectional(get_rnn())(x)
x = get_rnn(False)(x)

x = RepeatVector(maxlen)(x)
x = get_rnn()(x)
x = get_rnn()(x)
x = TimeDistributed(Dense(output_vocab_size, activation='softmax'))(x)


# We can refer to the parts of the model before and after <tt>get_rnn(False)</tt> returns a vector as the encoder and decoder. The encoder has taken a sequence of embeddings and encoded it into a numerical vector that completely describes it's input, while the decoder transforms that vector into a new sequence.
# 
# Now we can fit our model

# In[21]:


model = Model(inp, x)


# In[22]:


model.compile(Adam(), 'sparse_categorical_crossentropy', metrics=['acc'])


# In[23]:


hist=model.fit(input_train, np.expand_dims(labels_train,-1), 
          validation_data=[input_test, np.expand_dims(labels_test,-1)], 
          batch_size=64, **parms, nb_epoch=3)


# In[ ]:


hist.history['val_loss']


# To evaluate, we don't want to know what percentage of letters are correct but what percentage of words are.

# In[ ]:


def eval_keras(input):
    preds = model.predict(input, batch_size=128)
    predict = np.argmax(preds, axis = 2)
    return (np.mean([all(real==p) for real, p in zip(labels_test, predict)]), predict)


# The accuracy isn't great.

# In[ ]:


acc, preds = eval_keras(input_test); acc


# In[51]:


def print_examples(preds):
    print("pronunciation".ljust(40), "real spelling".ljust(17), 
          "model spelling".ljust(17), "is correct")

    for index in range(20):
        ps = "-".join([phonemes[p] for p in input_test[index]]) 
        real = [letters[l] for l in labels_test[index]] 
        predict = [letters[l] for l in preds[index]]
        print (ps.split("-_")[0].ljust(40), "".join(real).split("_")[0].ljust(17),
            "".join(predict).split("_")[0].ljust(17), str(real == predict))


# We can see that sometimes the mistakes are completely reasonable, occasionally they're totally off. This tends to happen with the longer words that have large phoneme sequences.
# 
# That's understandable; we'd expect larger sequences to lose more information in an encoding.

# In[52]:


print_examples(preds)


# ### Attention model

# This graph demonstrates the accuracy decay for a nueral translation task. With an encoding/decoding technique, larger input sequences result in less accuracy.
# 
# <img src="https://smerity.com/media/images/articles/2016/bahdanau_attn.png" width="600">
# 
# This can be mitigated using an attentional model.

# In[62]:


import attention_wrapper; importlib.reload(attention_wrapper)
from attention_wrapper import Attention


# The attentional model doesn't encode into a single vector, but rather a sequence of vectors. The decoder then at every point is passing through this sequence. For example, after the bi-directional RNN we have 16 vectors corresponding to each phoneme's output state. Each output state describes how each phoneme relates between the other phonemes before and after it. After going through more RNN's, our goal is to transform this sequence into a vector of length 15 so we can classify into characters. 
# 
# A smart way to take a weighted average of the 16 vectors for each of the 15 outputs, where each set of weights is unique to the output. For example, if character 1 only needs information from the first phoneme vector, that weight might be 1 and the others 0; if it needed information from the 1st and 2nd equally, those two might be 0.5 each.
# 
# The weights for combining all the input states to produce specific outputs can be learned using an attentional model; we update the weights using SGD, and train it jointly with the encoder/decoder. Once we have the outputs, we can classify the character using softmax as usual.

# Notice below we do not have an RNN that returns a flat vector as we did before; we have a sequence of vectors as desired. We can then pass a sequence of encoded states into the our custom <tt>Attention</tt> model.
# 
# This attention model also uses a technique called teacher forcing; in addition to passing the encoded hidden state, we also pass the correct answer for the previous time period. We give this information to the model because it makes it easier to train. In the beginning of training, the model will get most things wrong, and if your earlier character predictions are wrong then your later ones will likely be as well. Teacher forcing allows the model to still learn how to predict later characters, even if the earlier characters were all wrong.

# In[66]:


inp = Input((maxlen_p,))
inp_dec = Input((maxlen,))
emb_dec = Embedding(output_vocab_size, 120)(inp_dec)
emb_dec = Dense(dim)(emb_dec)

x = Embedding(input_vocab_size, 120)(inp)
x = Bidirectional(get_rnn())(x)
x = get_rnn()(x)
x = get_rnn()(x)
x = Attention(get_rnn, 3)([x, emb_dec])
x = TimeDistributed(Dense(output_vocab_size, activation='softmax'))(x)


# We can now train, passing in the decoder inputs as well for teacher forcing.

# In[67]:


model = Model([inp, inp_dec], x)
model.compile(Adam(), 'sparse_categorical_crossentropy', metrics=['acc'])


# In[68]:


hist=model.fit([input_train, dec_input_train], np.expand_dims(labels_train,-1), 
          validation_data=[[input_test, dec_input_test], np.expand_dims(labels_test,-1)], 
          batch_size=64, **parms, nb_epoch=3)


# In[25]:


hist.history['val_loss']


# In[998]:


K.set_value(model.optimizer.lr, 1e-4)


# In[999]:


hist=model.fit([input_train, dec_input_train], np.expand_dims(labels_train,-1), 
          validation_data=[[input_test, dec_input_test], np.expand_dims(labels_test,-1)], 
          batch_size=64, **parms, nb_epoch=5)


# In[1000]:


np.array(hist.history['val_loss'])


# In[1001]:


def eval_keras():
    preds = model.predict([input_test, dec_input_test], batch_size=128)
    predict = np.argmax(preds, axis = 2)
    return (np.mean([all(real==p) for real, p in zip(labels_test, predict)]), predict)


# Better accuracy!

# In[895]:


acc, preds = eval_keras(); acc


# This model is certainly performing better with longer words. The mistakes it's making are reasonable, and it even succesfully formed the word "partisanship".

# In[896]:


print("pronunciation".ljust(40), "real spelling".ljust(17), 
      "model spelling".ljust(17), "is correct")

for index in range(20):
    ps = "-".join([phonemes[p] for p in input_test[index]]) 
    real = [letters[l] for l in labels_test[index]] 
    predict = [letters[l] for l in preds[index]]
    print (ps.split("-_")[0].ljust(40), "".join(real).split("_")[0].ljust(17),
        "".join(predict).split("_")[0].ljust(17), str(real == predict))


# ## Test code for the attention layer

# In[301]:


nb_samples, nb_time, input_dim, output_dim = (64, 4, 32, 48)


# In[302]:


x = tf.placeholder(np.float32, (nb_samples, nb_time, input_dim))


# In[303]:


xr = K.reshape(x,(-1,nb_time,1,input_dim))


# In[304]:


W1 = tf.placeholder(np.float32, (input_dim, input_dim)); W1.shape


# In[305]:


W1r = K.reshape(W1, (1, input_dim, input_dim))


# In[306]:


W1r2 = K.reshape(W1, (1, 1, input_dim, input_dim))


# In[307]:


xW1 = K.conv1d(x,W1r,border_mode='same'); xW1.shape


# In[308]:


xW12 = K.conv2d(xr,W1r2,border_mode='same'); xW12.shape


# In[251]:


xW2 = K.dot(x, W1)


# In[245]:


x1 = np.random.normal(size=(nb_samples, nb_time, input_dim))


# In[246]:


w1 = np.random.normal(size=(input_dim, input_dim))


# In[248]:


res = sess.run(xW1, {x:x1, W1:w1})


# In[252]:


res2 = sess.run(xW2, {x:x1, W1:w1})


# In[253]:


np.allclose(res, res2)


# In[283]:


W2 = tf.placeholder(np.float32, (output_dim, input_dim)); W2.shape


# In[295]:


h = tf.placeholder(np.float32, (nb_samples, output_dim))


# In[296]:


hW2 = K.dot(h,W2); hW2.shape


# In[297]:


hW2 = K.reshape(hW2,(-1,1,1,input_dim)); hW2.shape

