
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

from fastai.imports import *
from fastai.torch_imports import *
from fastai.core import *
from fastai.model import fit
from fastai.dataset import *

import torchtext
from torchtext import vocab, data
from torchtext.datasets import language_modeling

from fastai.rnn_reg import *
from fastai.rnn_train import *
from fastai.nlp import *
from fastai.lm_rnn import *


# ## Language modeling

# ### Data

# In[2]:


PATH='data/wikitext-2/'
get_ipython().run_line_magic('ls', '{PATH}')


# In[3]:


get_ipython().system('head -5 {PATH}wiki.train.tokens')


# In[4]:


get_ipython().system('wc -lwc {PATH}wiki.train.tokens')


# In[5]:


get_ipython().system('wc -lwc {PATH}wiki.valid.tokens')


# In[6]:


TEXT = data.Field(lower=True)
FILES = dict(train='wiki.train.tokens', validation='wiki.valid.tokens', test='wiki.test.tokens')
bs,bptt = 80,70
md = LanguageModelData(PATH, TEXT, **FILES, bs=bs, bptt=bptt, min_freq=10)
len(md.trn_dl), md.nt, len(md.trn_ds), len(md.trn_ds[0].text)


# In[7]:


#md.trn_ds[0].text[:12], next(iter(md.trn_dl))


# ### Train

# In[9]:


em_sz = 200
nh = 500
nl = 3
learner = md.get_model(SGD_Momentum(0.7), bs, em_sz, nh, nl)
reg_fn=partial(seq2seq_reg, alpha=2, beta=1)


# In[10]:


clip=0.3
learner.fit(10, 1, wds=1e-6, reg_fn=reg_fn, clip=clip)


# In[11]:


learner.fit(10, 6, wds=1e-6, reg_fn=reg_fn, cycle_len=1, cycle_mult=2, clip=clip)


# In[12]:


learner.save('lm_420')


# In[13]:


learner.fit(10, 6, wds=1e-6, reg_fn=reg_fn, cycle_len=1, cycle_mult=2, clip=clip)


# In[14]:


learner.save('lm_419')


# In[15]:


learner.fit(10, 6, wds=1e-6, reg_fn=reg_fn, cycle_len=1, cycle_mult=2, clip=clip)


# In[16]:


learner.save('lm_418')


# In[17]:


math.exp(4.17)


# ### Test

# In[27]:


m=learner.model
s=[""". <eos> The game began development in 2010 , carrying over a large portion of the work 
done on Valkyria Chronicles II . While it retained the standard features of """.split()]
t=TEXT.numericalize(s)

m[0].bs=1
m.reset(False)
res,*_ = m(t)


# In[28]:


nexts = torch.topk(res[-1], 10)[1]
[TEXT.vocab.itos[o] for o in to_np(nexts)]


# In[29]:


for i in range(20):
    n=res[-1].topk(2)[1]
    n = n[1] if n.data[0]==0 else n[0]
    print(TEXT.vocab.itos[n.data[0]], end=' ')
    res,*_ = m(n[0].unsqueeze(0))


# ### End
