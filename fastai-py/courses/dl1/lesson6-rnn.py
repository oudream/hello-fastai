
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

from fastai.io import *
from fastai.conv_learner import *

from fastai.column_data import *


# ## Setup

# We're going to download the collected works of Nietzsche to use as our data for this class.

# In[2]:


PATH='data/nietzsche/'


# In[3]:


get_data("https://s3.amazonaws.com/text-datasets/nietzsche.txt", f'{PATH}nietzsche.txt')
text = open(f'{PATH}nietzsche.txt').read()
print('corpus length:', len(text))


# In[4]:


text[:400]


# In[5]:


chars = sorted(list(set(text)))
vocab_size = len(chars)+1
print('total chars:', vocab_size)


# Sometimes it's useful to have a zero value in the dataset, e.g. for padding

# In[6]:


chars.insert(0, "\0")

''.join(chars[1:-6])


# Map from chars to indices and back again

# In[7]:


char_indices = {c: i for i, c in enumerate(chars)}
indices_char = {i: c for i, c in enumerate(chars)}


# *idx* will be the data we use from now on - it simply converts all the characters to their index (based on the mapping above)

# In[8]:


idx = [char_indices[c] for c in text]

idx[:10]


# In[9]:


''.join(indices_char[i] for i in idx[:70])


# ## Three char model

# ### Create inputs

# Create a list of every 4th character, starting at the 0th, 1st, 2nd, then 3rd characters

# In[12]:


cs=3
c1_dat = [idx[i]   for i in range(0, len(idx)-cs, cs)]
c2_dat = [idx[i+1] for i in range(0, len(idx)-cs, cs)]
c3_dat = [idx[i+2] for i in range(0, len(idx)-cs, cs)]
c4_dat = [idx[i+3] for i in range(0, len(idx)-cs, cs)]


# Our inputs

# In[13]:


x1 = np.stack(c1_dat)
x2 = np.stack(c2_dat)
x3 = np.stack(c3_dat)


# Our output

# In[14]:


y = np.stack(c4_dat)


# The first 4 inputs and outputs

# In[14]:


x1[:4], x2[:4], x3[:4]


# In[15]:


y[:4]


# In[16]:


x1.shape, y.shape


# ### Create and train model

# Pick a size for our hidden state

# In[15]:


n_hidden = 256


# The number of latent factors to create (i.e. the size of the embedding matrix)

# In[16]:


n_fac = 42


# In[30]:


class Char3Model(nn.Module):
    def __init__(self, vocab_size, n_fac):
        super().__init__()
        self.e = nn.Embedding(vocab_size, n_fac)

        # The 'green arrow' from our diagram - the layer operation from input to hidden
        self.l_in = nn.Linear(n_fac, n_hidden)

        # The 'orange arrow' from our diagram - the layer operation from hidden to hidden
        self.l_hidden = nn.Linear(n_hidden, n_hidden)
        
        # The 'blue arrow' from our diagram - the layer operation from hidden to output
        self.l_out = nn.Linear(n_hidden, vocab_size)
        
    def forward(self, c1, c2, c3):
        in1 = F.relu(self.l_in(self.e(c1)))
        in2 = F.relu(self.l_in(self.e(c2)))
        in3 = F.relu(self.l_in(self.e(c3)))
        
        h = V(torch.zeros(in1.size()).cuda())
        h = F.tanh(self.l_hidden(h+in1))
        h = F.tanh(self.l_hidden(h+in2))
        h = F.tanh(self.l_hidden(h+in3))
        
        return F.log_softmax(self.l_out(h))


# In[20]:


md = ColumnarModelData.from_arrays('.', [-1], np.stack([x1,x2,x3], axis=1), y, bs=512)


# In[21]:


m = Char3Model(vocab_size, n_fac).cuda()


# In[22]:


it = iter(md.trn_dl)
*xs,yt = next(it)
t = m(*V(xs))


# In[191]:


opt = optim.Adam(m.parameters(), 1e-2)


# In[192]:


fit(m, md, 1, opt, F.nll_loss)


# In[193]:


set_lrs(opt, 0.001)


# In[194]:


fit(m, md, 1, opt, F.nll_loss)


# ### Test model

# In[195]:


def get_next(inp):
    idxs = T(np.array([char_indices[c] for c in inp]))
    p = m(*VV(idxs))
    i = np.argmax(to_np(p))
    return chars[i]


# In[196]:


get_next('y. ')


# In[197]:


get_next('ppl')


# In[198]:


get_next(' th')


# In[199]:


get_next('and')


# ## Our first RNN!

# ### Create inputs

# This is the size of our unrolled RNN.

# In[16]:


cs=8


# For each of 0 through 7, create a list of every 8th character with that starting point. These will be the 8 inputs to our model.

# In[17]:


c_in_dat = [[idx[i+j] for i in range(cs)] for j in range(len(idx)-cs)]


# Then create a list of the next character in each of these series. This will be the labels for our model.

# In[18]:


c_out_dat = [idx[j+cs] for j in range(len(idx)-cs)]


# In[66]:


xs = np.stack(c_in_dat, axis=0)


# In[67]:


xs.shape


# In[68]:


y = np.stack(c_out_dat)


# So each column below is one series of 8 characters from the text.

# In[69]:


xs[:cs,:cs]


# ...and this is the next character after each sequence.

# In[70]:


y[:cs]


# ### Create and train model

# In[71]:


val_idx = get_cv_idxs(len(idx)-cs-1)


# In[72]:


md = ColumnarModelData.from_arrays('.', val_idx, xs, y, bs=512)


# In[81]:


class CharLoopModel(nn.Module):
    # This is an RNN!
    def __init__(self, vocab_size, n_fac):
        super().__init__()
        self.e = nn.Embedding(vocab_size, n_fac)
        self.l_in = nn.Linear(n_fac, n_hidden)
        self.l_hidden = nn.Linear(n_hidden, n_hidden)
        self.l_out = nn.Linear(n_hidden, vocab_size)
        
    def forward(self, *cs):
        bs = cs[0].size(0)
        h = V(torch.zeros(bs, n_hidden).cuda())
        for c in cs:
            inp = F.relu(self.l_in(self.e(c)))
            h = F.tanh(self.l_hidden(h+inp))
        
        return F.log_softmax(self.l_out(h), dim=-1)


# In[82]:


m = CharLoopModel(vocab_size, n_fac).cuda()
opt = optim.Adam(m.parameters(), 1e-2)


# In[83]:


fit(m, md, 1, opt, F.nll_loss)


# In[84]:


set_lrs(opt, 0.001)


# In[85]:


fit(m, md, 1, opt, F.nll_loss)


# In[92]:


class CharLoopConcatModel(nn.Module):
    def __init__(self, vocab_size, n_fac):
        super().__init__()
        self.e = nn.Embedding(vocab_size, n_fac)
        self.l_in = nn.Linear(n_fac+n_hidden, n_hidden)
        self.l_hidden = nn.Linear(n_hidden, n_hidden)
        self.l_out = nn.Linear(n_hidden, vocab_size)
        
    def forward(self, *cs):
        bs = cs[0].size(0)
        h = V(torch.zeros(bs, n_hidden).cuda())
        for c in cs:
            inp = torch.cat((h, self.e(c)), 1)
            inp = F.relu(self.l_in(inp))
            h = F.tanh(self.l_hidden(inp))
        
        return F.log_softmax(self.l_out(h), dim=-1)


# In[93]:


m = CharLoopConcatModel(vocab_size, n_fac).cuda()
opt = optim.Adam(m.parameters(), 1e-3)


# In[94]:


it = iter(md.trn_dl)
*xs,yt = next(it)
t = m(*V(xs))


# In[95]:


fit(m, md, 1, opt, F.nll_loss)


# In[96]:


set_lrs(opt, 1e-4)


# In[97]:


fit(m, md, 1, opt, F.nll_loss)


# ### Test model

# In[98]:


def get_next(inp):
    idxs = T(np.array([char_indices[c] for c in inp]))
    p = m(*VV(idxs))
    i = np.argmax(to_np(p))
    return chars[i]


# In[99]:


get_next('for thos')


# In[100]:


get_next('part of ')


# In[101]:


get_next('queens a')


# ## RNN with pytorch

# In[108]:


class CharRnn(nn.Module):
    def __init__(self, vocab_size, n_fac):
        super().__init__()
        self.e = nn.Embedding(vocab_size, n_fac)
        self.rnn = nn.RNN(n_fac, n_hidden)
        self.l_out = nn.Linear(n_hidden, vocab_size)
        
    def forward(self, *cs):
        bs = cs[0].size(0)
        h = V(torch.zeros(1, bs, n_hidden))
        inp = self.e(torch.stack(cs))
        outp,h = self.rnn(inp, h)
        
        return F.log_softmax(self.l_out(outp[-1]), dim=-1)


# In[109]:


m = CharRnn(vocab_size, n_fac).cuda()
opt = optim.Adam(m.parameters(), 1e-3)


# In[110]:


it = iter(md.trn_dl)
*xs,yt = next(it)


# In[111]:


t = m.e(V(torch.stack(xs)))
t.size()


# In[112]:


ht = V(torch.zeros(1, 512,n_hidden))
outp, hn = m.rnn(t, ht)
outp.size(), hn.size()


# In[113]:


t = m(*V(xs)); t.size()


# In[114]:


fit(m, md, 4, opt, F.nll_loss)


# In[115]:


set_lrs(opt, 1e-4)


# In[116]:


fit(m, md, 2, opt, F.nll_loss)


# ### Test model

# In[117]:


def get_next(inp):
    idxs = T(np.array([char_indices[c] for c in inp]))
    p = m(*VV(idxs))
    i = np.argmax(to_np(p))
    return chars[i]


# In[118]:


get_next('for thos')


# In[119]:


def get_next_n(inp, n):
    res = inp
    for i in range(n):
        c = get_next(inp)
        res += c
        inp = inp[1:]+c
    return res


# In[120]:


get_next_n('for thos', 40)


# ## Multi-output model

# ### Setup

# Let's take non-overlapping sets of characters this time

# In[19]:


c_in_dat = [[idx[i+j] for i in range(cs)] for j in range(0, len(idx)-cs-1, cs)]


# Then create the exact same thing, offset by 1, as our labels

# In[20]:


c_out_dat = [[idx[i+j] for i in range(cs)] for j in range(1, len(idx)-cs, cs)]


# In[21]:


xs = np.stack(c_in_dat)
xs.shape


# In[22]:


ys = np.stack(c_out_dat)
ys.shape


# In[23]:


xs[:cs,:cs]


# In[24]:


ys[:cs,:cs]


# ### Create and train model

# In[127]:


val_idx = get_cv_idxs(len(xs)-cs-1)


# In[128]:


md = ColumnarModelData.from_arrays('.', val_idx, xs, ys, bs=512)


# In[133]:


class CharSeqRnn(nn.Module):
    def __init__(self, vocab_size, n_fac):
        super().__init__()
        self.e = nn.Embedding(vocab_size, n_fac)
        self.rnn = nn.RNN(n_fac, n_hidden)
        self.l_out = nn.Linear(n_hidden, vocab_size)
        
    def forward(self, *cs):
        bs = cs[0].size(0)
        h = V(torch.zeros(1, bs, n_hidden))
        inp = self.e(torch.stack(cs))
        outp,h = self.rnn(inp, h)
        return F.log_softmax(self.l_out(outp), dim=-1)


# In[134]:


m = CharSeqRnn(vocab_size, n_fac).cuda()
opt = optim.Adam(m.parameters(), 1e-3)


# In[135]:


it = iter(md.trn_dl)
*xst,yt = next(it)


# In[136]:


def nll_loss_seq(inp, targ):
    sl,bs,nh = inp.size()
    targ = targ.transpose(0,1).contiguous().view(-1)
    return F.nll_loss(inp.view(-1,nh), targ)


# In[137]:


fit(m, md, 4, opt, nll_loss_seq)


# In[138]:


set_lrs(opt, 1e-4)


# In[139]:


fit(m, md, 1, opt, nll_loss_seq)


# ### Identity init!

# In[140]:


m = CharSeqRnn(vocab_size, n_fac).cuda()
opt = optim.Adam(m.parameters(), 1e-2)


# In[141]:


m.rnn.weight_hh_l0.data.copy_(torch.eye(n_hidden))


# In[142]:


fit(m, md, 4, opt, nll_loss_seq)


# In[143]:


set_lrs(opt, 1e-3)


# In[144]:


fit(m, md, 4, opt, nll_loss_seq)


# ## Stateful model

# ### Setup

# In[15]:


from torchtext import vocab, data

from fastai.nlp import *
from fastai.lm_rnn import *

PATH='data/nietzsche/'

TRN_PATH = 'trn/'
VAL_PATH = 'val/'
TRN = f'{PATH}{TRN_PATH}'
VAL = f'{PATH}{VAL_PATH}'

# Note: The student needs to practice her shell skills and prepare her own dataset before proceeding:
# - trn/trn.txt (first 80% of nietzsche.txt)
# - val/val.txt (last 20% of nietzsche.txt)

get_ipython().run_line_magic('ls', '{PATH}')


# In[16]:


get_ipython().run_line_magic('ls', '{PATH}trn')


# In[18]:


TEXT = data.Field(lower=True, tokenize=list)
bs=64; bptt=8; n_fac=42; n_hidden=256

FILES = dict(train=TRN_PATH, validation=VAL_PATH, test=VAL_PATH)
md = LanguageModelData.from_text_files(PATH, TEXT, **FILES, bs=bs, bptt=bptt, min_freq=3)

len(md.trn_dl), md.nt, len(md.trn_ds), len(md.trn_ds[0].text)


# ### RNN

# In[21]:


class CharSeqStatefulRnn(nn.Module):
    def __init__(self, vocab_size, n_fac, bs):
        self.vocab_size = vocab_size
        super().__init__()
        self.e = nn.Embedding(vocab_size, n_fac)
        self.rnn = nn.RNN(n_fac, n_hidden)
        self.l_out = nn.Linear(n_hidden, vocab_size)
        self.init_hidden(bs)
        
    def forward(self, cs):
        bs = cs[0].size(0)
        if self.h.size(1) != bs: self.init_hidden(bs)
        outp,h = self.rnn(self.e(cs), self.h)
        self.h = repackage_var(h)
        return F.log_softmax(self.l_out(outp), dim=-1).view(-1, self.vocab_size)
    
    def init_hidden(self, bs): self.h = V(torch.zeros(1, bs, n_hidden))


# In[22]:


m = CharSeqStatefulRnn(md.nt, n_fac, 512).cuda()
opt = optim.Adam(m.parameters(), 1e-3)


# In[23]:


fit(m, md, 4, opt, F.nll_loss)


# In[24]:


set_lrs(opt, 1e-4)

fit(m, md, 4, opt, F.nll_loss)


# ### RNN loop

# In[9]:


# From the pytorch source

def RNNCell(input, hidden, w_ih, w_hh, b_ih, b_hh):
    return F.tanh(F.linear(input, w_ih, b_ih) + F.linear(hidden, w_hh, b_hh))


# In[12]:


class CharSeqStatefulRnn2(nn.Module):
    def __init__(self, vocab_size, n_fac, bs):
        super().__init__()
        self.vocab_size = vocab_size
        self.e = nn.Embedding(vocab_size, n_fac)
        self.rnn = nn.RNNCell(n_fac, n_hidden)
        self.l_out = nn.Linear(n_hidden, vocab_size)
        self.init_hidden(bs)
        
    def forward(self, cs):
        bs = cs[0].size(0)
        if self.h.size(1) != bs: self.init_hidden(bs)
        outp = []
        o = self.h
        for c in cs: 
            o = self.rnn(self.e(c), o)
            outp.append(o)
        outp = self.l_out(torch.stack(outp))
        self.h = repackage_var(o)
        return F.log_softmax(outp, dim=-1).view(-1, self.vocab_size)
    
    def init_hidden(self, bs): self.h = V(torch.zeros(1, bs, n_hidden))


# In[13]:


m = CharSeqStatefulRnn2(md.nt, n_fac, 512).cuda()
opt = optim.Adam(m.parameters(), 1e-3)


# In[8]:


fit(m, md, 4, opt, F.nll_loss)


# ### GRU

# In[18]:


class CharSeqStatefulGRU(nn.Module):
    def __init__(self, vocab_size, n_fac, bs):
        super().__init__()
        self.vocab_size = vocab_size
        self.e = nn.Embedding(vocab_size, n_fac)
        self.rnn = nn.GRU(n_fac, n_hidden)
        self.l_out = nn.Linear(n_hidden, vocab_size)
        self.init_hidden(bs)
        
    def forward(self, cs):
        bs = cs[0].size(0)
        if self.h.size(1) != bs: self.init_hidden(bs)
        outp,h = self.rnn(self.e(cs), self.h)
        self.h = repackage_var(h)
        return F.log_softmax(self.l_out(outp), dim=-1).view(-1, self.vocab_size)
    
    def init_hidden(self, bs): self.h = V(torch.zeros(1, bs, n_hidden))


# In[ ]:


# From the pytorch source code - for reference

def GRUCell(input, hidden, w_ih, w_hh, b_ih, b_hh):
    gi = F.linear(input, w_ih, b_ih)
    gh = F.linear(hidden, w_hh, b_hh)
    i_r, i_i, i_n = gi.chunk(3, 1)
    h_r, h_i, h_n = gh.chunk(3, 1)

    resetgate = F.sigmoid(i_r + h_r)
    inputgate = F.sigmoid(i_i + h_i)
    newgate = F.tanh(i_n + resetgate * h_n)
    return newgate + inputgate * (hidden - newgate)


# In[27]:


m = CharSeqStatefulGRU(md.nt, n_fac, 512).cuda()

opt = optim.Adam(m.parameters(), 1e-3)


# In[29]:


fit(m, md, 6, opt, F.nll_loss)


# In[30]:


set_lrs(opt, 1e-4)


# In[31]:


fit(m, md, 3, opt, F.nll_loss)


# ### Putting it all together: LSTM

# In[12]:


from fastai import sgdr

n_hidden=512


# In[22]:


class CharSeqStatefulLSTM(nn.Module):
    def __init__(self, vocab_size, n_fac, bs, nl):
        super().__init__()
        self.vocab_size,self.nl = vocab_size,nl
        self.e = nn.Embedding(vocab_size, n_fac)
        self.rnn = nn.LSTM(n_fac, n_hidden, nl, dropout=0.5)
        self.l_out = nn.Linear(n_hidden, vocab_size)
        self.init_hidden(bs)
        
    def forward(self, cs):
        bs = cs[0].size(0)
        if self.h[0].size(1) != bs: self.init_hidden(bs)
        outp,h = self.rnn(self.e(cs), self.h)
        self.h = repackage_var(h)
        return F.log_softmax(self.l_out(outp), dim=-1).view(-1, self.vocab_size)
    
    def init_hidden(self, bs):
        self.h = (V(torch.zeros(self.nl, bs, n_hidden)),
                  V(torch.zeros(self.nl, bs, n_hidden)))


# In[23]:


m = CharSeqStatefulLSTM(md.nt, n_fac, 512, 2).cuda()
lo = LayerOptimizer(optim.Adam, m, 1e-2, 1e-5)


# In[18]:


os.makedirs(f'{PATH}models', exist_ok=True)


# In[19]:


fit(m, md, 2, lo.opt, F.nll_loss)


# In[20]:


on_end = lambda sched, cycle: save_model(m, f'{PATH}models/cyc_{cycle}')
cb = [CosAnneal(lo, len(md.trn_dl), cycle_mult=2, on_cycle_end=on_end)]
fit(m, md, 2**4-1, lo.opt, F.nll_loss, callbacks=cb)


# In[44]:


on_end = lambda sched, cycle: save_model(m, f'{PATH}models/cyc_{cycle}')
cb = [CosAnneal(lo, len(md.trn_dl), cycle_mult=2, on_cycle_end=on_end)]
fit(m, md, 2**6-1, lo.opt, F.nll_loss, callbacks=cb)


# ### Test

# In[45]:


def get_next(inp):
    idxs = TEXT.numericalize(inp)
    p = m(VV(idxs.transpose(0,1)))
    r = torch.multinomial(p[-1].exp(), 1)
    return TEXT.vocab.itos[to_np(r)[0]]


# In[46]:


get_next('for thos')


# In[47]:


def get_next_n(inp, n):
    res = inp
    for i in range(n):
        c = get_next(inp)
        res += c
        inp = inp[1:]+c
    return res


# In[50]:


print(get_next_n('for thos', 400))

