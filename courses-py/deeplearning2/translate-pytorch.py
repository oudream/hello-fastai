
# coding: utf-8

# # Translating French to English with Pytorch

# In[53]:


get_ipython().magic(u'matplotlib inline')
import re, pickle, collections, bcolz, numpy as np, keras, sklearn, math, operator


# In[4]:


from gensim.models import word2vec

import torch, torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F


# In[5]:


path='/data/datasets/fr-en-109-corpus/'
dpath = 'data/translate/'


# ## Prepare corpus

# The French-English parallel corpus can be downloaded from http://www.statmt.org/wmt10/training-giga-fren.tar. It was created by Chris Callison-Burch, who crawled millions of web pages and then used 'a set of simple heuristics to transform French URLs onto English URLs (i.e. replacing "fr" with "en" and about 40 other hand-written rules), and assume that these documents are translations of each other'.

# In[6]:


fname=path+'giga-fren.release2.fixed'
en_fname = fname+'.en'
fr_fname = fname+'.fr'


# To make this problem a little simpler so we can train our model more quickly, we'll just learn to translate questions that begin with 'Wh' (e.g. what, why, where which). Here are our regexps that filter the sentences we want.

# In[11]:


re_eq = re.compile('^(Wh[^?.!]+\?)')
re_fq = re.compile('^([^?.!]+\?)')

lines = ((re_eq.search(eq), re_fq.search(fq)) 
         for eq, fq in zip(open(en_fname), open(fr_fname)))

qs = [(e.group(), f.group()) for e,f in lines if e and f]; len(qs)


# In[12]:


qs[:6]


# Because it takes a while to load the data, we save the results to make it easier to load in later.

# In[17]:


pickle.dump(qs, open(dpath+'fr-en-qs.pkl', 'wb'))


# In[7]:


qs = pickle.load(open(dpath+'fr-en-qs.pkl', 'rb'))


# In[8]:


en_qs, fr_qs = zip(*qs)


# Because we are translating at word level, we need to tokenize the text first. (Note that it is also possible to translate at character level, which doesn't require tokenizing.) There are many tokenizers available, but we found we got best results using these simple heuristics.

# In[9]:


re_apos = re.compile(r"(\w)'s\b")         # make 's a separate word
re_mw_punc = re.compile(r"(\w[’'])(\w)")  # other ' in a word creates 2 words
re_punc = re.compile("([\"().,;:/_?!—])") # add spaces around punctuation
re_mult_space = re.compile(r"  *")        # replace multiple spaces with just one

def simple_toks(sent):
    sent = re_apos.sub(r"\1 's", sent)
    sent = re_mw_punc.sub(r"\1 \2", sent)
    sent = re_punc.sub(r" \1 ", sent).replace('-', ' ')
    sent = re_mult_space.sub(' ', sent)
    return sent.lower().split()


# In[10]:


fr_qtoks = list(map(simple_toks, fr_qs)); fr_qtoks[:4]


# In[11]:


en_qtoks = list(map(simple_toks, en_qs)); en_qtoks[:4]


# In[12]:


simple_toks("Rachel's baby is cuter than other's.")


# Special tokens used to pad the end of sentences, and to mark the start of a sentence.

# In[13]:


PAD = 0; SOS = 1


# Enumerate the unique words (*vocab*) in the corpus, and also create the reverse map (word->index). Then use this mapping to encode every sentence as a list of int indices.

# In[14]:


def toks2ids(sents):
    voc_cnt = collections.Counter(t for sent in sents for t in sent)
    vocab = sorted(voc_cnt, key=voc_cnt.get, reverse=True)
    vocab.insert(PAD, "<PAD>")
    vocab.insert(SOS, "<SOS>")
    w2id = {w:i for i,w in enumerate(vocab)}
    ids = [[w2id[t] for t in sent] for sent in sents]
    return ids, vocab, w2id, voc_cnt


# In[15]:


fr_ids, fr_vocab, fr_w2id, fr_counts = toks2ids(fr_qtoks)
en_ids, en_vocab, en_w2id, en_counts = toks2ids(en_qtoks)


# ## Word vectors

# Stanford's GloVe word vectors can be downloaded from https://nlp.stanford.edu/projects/glove/ (in the code below we have preprocessed them into a bcolz array). We use these because each individual word has a single word vector, which is what we need for translation. Word2vec, on the other hand, often uses multi-word phrases.

# In[16]:


def load_glove(loc):
    return (bcolz.open(loc+'.dat')[:],
        pickle.load(open(loc+'_words.pkl','rb'), encoding='latin1'),
        pickle.load(open(loc+'_idx.pkl','rb'), encoding='latin1'))


# In[17]:


en_vecs, en_wv_word, en_wv_idx = load_glove('/data/datasets/nlp/glove/results/6B.100d')
en_w2v = {w: en_vecs[en_wv_idx[w]] for w in en_wv_word}
n_en_vec, dim_en_vec = en_vecs.shape


# In[87]:


en_w2v['king']


# For French word vectors, we're using those from http://fauconnier.github.io/index.html

# In[18]:


w2v_path='/data/datasets/nlp/frWac_non_lem_no_postag_no_phrase_200_skip_cut100.bin'
fr_model = word2vec.Word2Vec.load_word2vec_format(w2v_path, binary=True)
fr_voc = fr_model.vocab
dim_fr_vec = 200


# We need to map each word index in our vocabs to their word vector. Not every word in our vocabs will be in our word vectors, since our tokenization approach won't be identical to the word vector creators - in these cases we simply create a random vector.

# In[19]:


def create_emb(w2v, targ_vocab, dim_vec):
    vocab_size = len(targ_vocab)
    emb = np.zeros((vocab_size, dim_vec))
    found=0

    for i, word in enumerate(targ_vocab):
        try: emb[i] = w2v[word]; found+=1
        except KeyError: emb[i] = np.random.normal(scale=0.6, size=(dim_vec,))

    return emb, found


# In[20]:


en_embs, found = create_emb(en_w2v, en_vocab, dim_en_vec); en_embs.shape, found


# In[21]:


fr_embs, found = create_emb(fr_model, fr_vocab, dim_fr_vec); fr_embs.shape, found


# ## Prep data

# Each sentence has to be of equal length. Keras has a convenient function `pad_sequences` to truncate and/or pad each sentence as required - even although we're not using keras for the neural net, we can still use any functions from it we need!

# In[22]:


from keras.preprocessing.sequence import pad_sequences

maxlen = 30
en_padded = pad_sequences(en_ids, maxlen, 'int64', "post", "post")
fr_padded = pad_sequences(fr_ids, maxlen, 'int64', "post", "post")
en_padded.shape, fr_padded.shape, en_embs.shape


# And of course we need to separate our training and test sets...

# In[30]:


from sklearn import model_selection
fr_train, fr_test, en_train, en_test = model_selection.train_test_split(
    fr_padded, en_padded, test_size=0.1)

[o.shape for o in (fr_train, fr_test, en_train, en_test)]


# Here's an example of a French and English sentence, after encoding and padding.

# In[31]:


fr_train[0], en_train[0]


# ## Model

# ### Basic encoder-decoder

# In[65]:


def long_t(arr): return Variable(torch.LongTensor(arr)).cuda()


# In[68]:


fr_emb_t = torch.FloatTensor(fr_embs).cuda()
en_emb_t = torch.FloatTensor(en_embs).cuda()


# In[56]:


def create_emb(emb_mat, non_trainable=False):
    output_size, emb_size = emb_mat.size()
    emb = nn.Embedding(output_size, emb_size)
    emb.load_state_dict({'weight': emb_mat})
    if non_trainable:
        for param in emb.parameters(): 
            param.requires_grad = False
    return emb, emb_size, output_size


# Turning a sequence into a representation can be done using an RNN (called the 'encoder'. This approach is useful because RNN's are able to keep track of state and memory, which is obviously important in forming a complete understanding of a sentence.
# * `bidirectional=True` passes the original sequence through an RNN, and the reversed sequence through a different RNN and concatenates the results. This allows us to look forward and backwards.
# * We do this because in language things that happen later often influence what came before (i.e. in Spanish, "el chico, la chica" means the boy, the girl; the word for "the" is determined by the gender of the subject, which comes after).

# In[57]:


class EncoderRNN(nn.Module):
    def __init__(self, embs, hidden_size, n_layers=2):
        super(EncoderRNN, self).__init__()
        self.emb, emb_size, output_size = create_emb(embs, True)
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(emb_size, hidden_size, batch_first=True, num_layers=n_layers)
#                          ,bidirectional=True)
        
    def forward(self, input, hidden):
        return self.gru(self.emb(input), hidden)

    def initHidden(self, batch_size):
        return Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))


# In[58]:


def encode(inp, encoder):
    batch_size, input_length = inp.size()
    hidden = encoder.initHidden(batch_size).cuda()
    enc_outputs, hidden = encoder(inp, hidden)
    return long_t([SOS]*batch_size), enc_outputs, hidden    


# Finally, we arrive at a vector representation of the sequence which captures everything we need to translate it. We feed this vector into more RNN's, which are trying to generate the labels. After this, we make a classification for what each word is in the output sequence.

# In[59]:


class DecoderRNN(nn.Module):
    def __init__(self, embs, hidden_size, n_layers=2):
        super(DecoderRNN, self).__init__()
        self.emb, emb_size, output_size = create_emb(embs)
        self.gru = nn.GRU(emb_size, hidden_size, batch_first=True, num_layers=n_layers)
        self.out = nn.Linear(hidden_size, output_size)
        
    def forward(self, inp, hidden):
        emb = self.emb(inp).unsqueeze(1)
        res, hidden = self.gru(emb, hidden)
        res = F.log_softmax(self.out(res[:,0]))
        return res, hidden


# This graph demonstrates the accuracy decay for a neural translation task. With an encoding/decoding technique, larger input sequences result in less accuracy.
# 
# <img src="https://smerity.com/media/images/articles/2016/bahdanau_attn.png" width="600">
# 
# This can be mitigated using an attentional model.

# ### Adding broadcasting to Pytorch

# Using *broadcasting* makes a lot of numerical programming far simpler. Here's a couple of examples, using numpy:

# In[32]:


v=np.array([1,2,3]); v, v.shape


# In[33]:


m=np.array([v,v*2,v*3]); m, m.shape


# In[34]:


m+v


# In[35]:


v1=np.expand_dims(v,-1); v1, v1.shape


# In[36]:


m+v1


# But Pytorch doesn't support broadcasting. So let's add it to the basic operators, and to a general tensor dot product:

# In[44]:


def unit_prefix(x, n=1):
    for i in range(n): x = x.unsqueeze(0)
    return x

def align(x, y, start_dim=2):
    xd, yd = x.dim(), y.dim()
    if xd > yd: y = unit_prefix(y, xd - yd)
    elif yd > xd: x = unit_prefix(x, yd - xd)

    xs, ys = list(x.size()), list(y.size())
    nd = len(ys)
    for i in range(start_dim, nd):
        td = nd-i-1
        if   ys[td]==1: ys[td] = xs[td]
        elif xs[td]==1: xs[td] = ys[td]
    return x.expand(*xs), y.expand(*ys)


# In[45]:


def aligned_op(x,y,f): return f(*align(x,y,0))

def add(x, y): return aligned_op(x, y, operator.add)
def sub(x, y): return aligned_op(x, y, operator.sub)
def mul(x, y): return aligned_op(x, y, operator.mul)
def div(x, y): return aligned_op(x, y, operator.truediv)


# In[46]:


def dot(x, y):
    assert(1<y.dim()<5)
    x, y = align(x, y)
    
    if y.dim() == 2: return x.mm(y)
    elif y.dim() == 3: return x.bmm(y)
    else:
        xs,ys = x.size(), y.size()
        res = torch.zeros(*(xs[:-1] + (ys[-1],)))
        for i in range(xs[0]): res[i].baddbmm_(x[i], (y[i]))
        return res


# Let's test!

# In[47]:


def Arr(*sz): return torch.randn(sz)/math.sqrt(sz[0])


# In[48]:


m = Arr(3, 2); m2 = Arr(4, 3)
v = Arr(2)
b = Arr(4,3,2); t = Arr(5,4,3,2)

mt,bt,tt = m.transpose(0,1), b.transpose(1,2), t.transpose(2,3)


# In[49]:


def check_eq(x,y): assert(torch.equal(x,y))


# In[50]:


check_eq(dot(m,mt),m.mm(mt))
check_eq(dot(v,mt), v.unsqueeze(0).mm(mt))
check_eq(dot(b,bt),b.bmm(bt))
check_eq(dot(b,mt),b.bmm(unit_prefix(mt).expand_as(bt)))


# In[51]:


exp = t.view(-1,3,2).bmm(tt.contiguous().view(-1,2,3)).view(5,4,3,3)
check_eq(dot(t,tt),exp)


# In[54]:


check_eq(add(m,v),m+unit_prefix(v).expand_as(m))
check_eq(add(v,m),m+unit_prefix(v).expand_as(m))
check_eq(add(m,t),t+unit_prefix(m,2).expand_as(t))
check_eq(sub(m,v),m-unit_prefix(v).expand_as(m))
check_eq(mul(m,v),m*unit_prefix(v).expand_as(m))
check_eq(div(m,v),m/unit_prefix(v).expand_as(m))


# ### Attentional model

# In[55]:


def Var(*sz): return Parameter(Arr(*sz)).cuda()


# In[60]:


class AttnDecoderRNN(nn.Module):
    def __init__(self, embs, hidden_size, n_layers=2, p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.emb, emb_size, output_size = create_emb(embs)
        self.W1 = Var(hidden_size, hidden_size)
        self.W2 = Var(hidden_size, hidden_size)
        self.W3 = Var(emb_size+hidden_size, hidden_size)
        self.b2 = Var(hidden_size)
        self.b3 = Var(hidden_size)
        self.V = Var(hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=2)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, inp, hidden, enc_outputs):
        emb_inp = self.emb(inp)
        w1e = dot(enc_outputs, self.W1)
        w2h = add(dot(hidden[-1], self.W2), self.b2).unsqueeze(1)
        u = F.tanh(add(w1e, w2h))
        a = mul(self.V,u).sum(2).squeeze(2)
        a = F.softmax(a).unsqueeze(2)
        Xa = mul(a, enc_outputs).sum(1)
        res = dot(torch.cat([emb_inp, Xa.squeeze(1)], 1), self.W3)
        res = add(res, self.b3).unsqueeze(0)
        res, hidden = self.gru(res, hidden)
        res = F.log_softmax(self.out(res.squeeze(0)))
        return res, hidden


# ### Attention testing

# Pytorch makes it easy to check intermediate results, when creating a custom architecture such as this one, since you can interactively run each function.

# In[66]:


def get_batch(x, y, batch_size=16):
    idxs = np.random.permutation(len(x))[:batch_size]
    return x[idxs], y[idxs]


# In[69]:


hidden_size = 128
fra, eng = get_batch(fr_train, en_train, 4)
inp = long_t(fra)
targ = long_t(eng)
emb, emb_size, output_size = create_emb(en_emb_t)
emb.cuda()
inp.size()


# In[70]:


W1 = Var(hidden_size, hidden_size)
W2 = Var(hidden_size, hidden_size)
W3 = Var(emb_size+hidden_size, hidden_size)
b2 = Var(1,hidden_size)
b3 = Var(1,hidden_size)
V = Var(1,1,hidden_size)
gru = nn.GRU(hidden_size, hidden_size, num_layers=2).cuda()
out = nn.Linear(hidden_size, output_size).cuda()


# In[73]:


dec_inputs, enc_outputs, hidden = encode(inp, encoder)
enc_outputs.size(), hidden.size()


# In[74]:


emb_inp = emb(dec_inputs); emb_inp.size()


# In[75]:


w1e = dot(enc_outputs, W1); w1e.size()


# In[76]:


w2h = dot(hidden[-1], W2)
w2h = (w2h+b2.expand_as(w2h)).unsqueeze(1); w2h.size()


# In[77]:


u = F.tanh(w1e + w2h.expand_as(w1e))
a = (V.expand_as(u)*u).sum(2).squeeze(2)
a = F.softmax(a).unsqueeze(2); a.size(),a.sum(1).squeeze(1)


# In[78]:


Xa = (a.expand_as(enc_outputs) * enc_outputs).sum(1); Xa.size()


# In[79]:


res = dot(torch.cat([emb_inp, Xa.squeeze(1)], 1), W3)
res = (res+b3.expand_as(res)).unsqueeze(0); res.size()


# In[80]:


res, hidden = gru(res, hidden); res.size(), hidden.size()


# In[81]:


res = F.log_softmax(out(res.squeeze(0))); res.size()


# ## Train

# Pytorch has limited functionality for training models automatically - you will generally have to write your own training loops. However, Pytorch makes it far easier to customize how this training is done, such as using *teacher forcing*.

# In[82]:


def train(inp, targ, encoder, decoder, enc_opt, dec_opt, crit):
    decoder_input, encoder_outputs, hidden = encode(inp, encoder)
    target_length = targ.size()[1]
    
    enc_opt.zero_grad(); dec_opt.zero_grad()
    loss = 0

    for di in range(target_length):
        decoder_output, hidden = decoder(decoder_input, hidden, encoder_outputs)
        decoder_input = targ[:, di]
        loss += crit(decoder_output, decoder_input)

    loss.backward()
    enc_opt.step(); dec_opt.step()
    return loss.data[0] / target_length


# In[83]:


def req_grad_params(o):
    return (p for p in o.parameters() if p.requires_grad)


# In[84]:


def trainEpochs(encoder, decoder, n_epochs, print_every=1000, lr=0.01):
    loss_total = 0 # Reset every print_every
    
    enc_opt = optim.RMSprop(req_grad_params(encoder), lr=lr)
    dec_opt = optim.RMSprop(decoder.parameters(), lr=lr)
    crit = nn.NLLLoss().cuda()
    
    for epoch in range(n_epochs):
        fra, eng = get_batch(fr_train, en_train, 64)
        inp = long_t(fra)
        targ = long_t(eng)
        loss = train(inp, targ, encoder, decoder, enc_opt, dec_opt, crit)
        loss_total += loss

        if epoch % print_every == print_every-1:
            print('%d %d%% %.4f' % (epoch, epoch / n_epochs * 100, loss_total / print_every))
            loss_total = 0


# ## Run

# In[85]:


hidden_size = 128
encoder = EncoderRNN(fr_emb_t, hidden_size).cuda()
decoder = AttnDecoderRNN(en_emb_t, hidden_size).cuda()


# In[148]:


trainEpochs(encoder, decoder, 10000, print_every=500, lr=0.005)


# ## Testing

# In[234]:


def evaluate(inp):
    decoder_input, encoder_outputs, hidden = encode(inp, encoder)
    target_length = maxlen

    decoded_words = []
    for di in range(target_length):
        decoder_output, hidden = decoder(decoder_input, hidden, encoder_outputs)
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni==PAD: break
        decoded_words.append(en_vocab[ni])
        decoder_input = long_t([ni])
    
    return decoded_words


# In[235]:


def sent2ids(sent):
    ids = [fr_w2id[t] for t in simple_toks(sent)]
    return pad_sequences([ids], maxlen, 'int64', "post", "post")


# In[236]:


def fr2en(sent): 
    ids = long_t(sent2ids(sent))
    trans = evaluate(ids)
    return ' '.join(trans)


# In[237]:


i=8
print(en_qs[i],fr_qs[i])
fr2en(fr_qs[i])

