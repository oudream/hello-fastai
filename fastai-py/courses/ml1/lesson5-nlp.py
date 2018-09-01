
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

from fastai.nlp import *
from sklearn.linear_model import LogisticRegression


# ## IMDB dataset and the sentiment classification task

# The [large movie review dataset](http://ai.stanford.edu/~amaas/data/sentiment/) contains a collection of 50,000 reviews from IMDB. The dataset contains an even number of positive and negative reviews. The authors considered only highly polarized reviews. A negative review has a score ≤ 4 out of 10, and a positive review has a score ≥ 7 out of 10. Neutral reviews are not included in the dataset. The dataset is divided into training and test sets. The training set is the same 25,000 labeled reviews.
# 
# The **sentiment classification task** consists of predicting the polarity (positive or negative) of a given text.
# 
# To get the dataset, in your terminal run the following commands:
# 
# `wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz`
# 
# `gunzip aclImdb_v1.tar.gz`
# 
# `tar -xvf aclImdb_v1.tar`

# ### Tokenizing and term document matrix creation

# In[2]:


PATH='data/aclImdb/'
names = ['neg','pos']


# In[3]:


get_ipython().run_line_magic('ls', '{PATH}')


# In[22]:


get_ipython().run_line_magic('ls', '{PATH}train')


# In[23]:


get_ipython().run_line_magic('ls', '{PATH}train/pos | head')


# In[24]:


trn,trn_y = texts_labels_from_folders(f'{PATH}train',names)
val,val_y = texts_labels_from_folders(f'{PATH}test',names)


# Here is the text of the first review

# In[25]:


trn[0]


# In[26]:


trn_y[0]


# [`CountVectorizer`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) converts a collection of text documents to a matrix of token counts (part of `sklearn.feature_extraction.text`).

# In[27]:


veczr = CountVectorizer(tokenizer=tokenize)


# `fit_transform(trn)` finds the vocabulary in the training set. It also transforms the training set into a term-document matrix. Since we have to apply the *same transformation* to your validation set, the second line uses just the method `transform(val)`. `trn_term_doc` and `val_term_doc` are sparse matrices. `trn_term_doc[i]` represents training document i and it contains a count of words for each document for each word in the vocabulary.

# In[28]:


trn_term_doc = veczr.fit_transform(trn)
val_term_doc = veczr.transform(val)


# In[29]:


trn_term_doc


# In[30]:


trn_term_doc[0]


# In[31]:


vocab = veczr.get_feature_names(); vocab[5000:5005]


# In[32]:


w0 = set([o.lower() for o in trn[0].split(' ')]); w0


# In[33]:


len(w0)


# In[34]:


veczr.vocabulary_['absurd']


# In[35]:


trn_term_doc[0,1297]


# In[36]:


trn_term_doc[0,5000]


# ## Naive Bayes

# We define the **log-count ratio** $r$ for each word $f$:
# 
# $r = \log \frac{\text{ratio of feature $f$ in positive documents}}{\text{ratio of feature $f$ in negative documents}}$
# 
# where ratio of feature $f$ in positive documents is the number of times a positive document has a feature divided by the number of positive documents.

# In[89]:


def pr(y_i):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)


# In[90]:


x=trn_term_doc
y=trn_y

r = np.log(pr(1)/pr(0))
b = np.log((y==1).mean() / (y==0).mean())


# Here is the formula for Naive Bayes.

# In[84]:


pre_preds = val_term_doc @ r.T + b
preds = pre_preds.T>0
(preds==val_y).mean()


# ...and binarized Naive Bayes.

# In[91]:


x=trn_term_doc.sign()
r = np.log(pr(1)/pr(0))

pre_preds = val_term_doc.sign() @ r.T + b
preds = pre_preds.T>0
(preds==val_y).mean()


# ### Logistic regression

# Here is how we can fit logistic regression where the features are the unigrams.

# In[22]:


m = LogisticRegression(C=1e8, dual=True)
m.fit(x, y)
preds = m.predict(val_term_doc)
(preds==val_y).mean()


# In[23]:


m = LogisticRegression(C=1e8, dual=True)
m.fit(trn_term_doc.sign(), y)
preds = m.predict(val_term_doc.sign())
(preds==val_y).mean()


# ...and the regularized version

# In[24]:


m = LogisticRegression(C=0.1, dual=True)
m.fit(x, y)
preds = m.predict(val_term_doc)
(preds==val_y).mean()


# In[25]:


m = LogisticRegression(C=0.1, dual=True)
m.fit(trn_term_doc.sign(), y)
preds = m.predict(val_term_doc.sign())
(preds==val_y).mean()


# ### Trigram with NB features

# Our next model is a version of logistic regression with Naive Bayes features described [here](https://www.aclweb.org/anthology/P12-2018). For every document we compute binarized features as described above, but this time we use bigrams and trigrams too. Each feature is a log-count ratio. A logistic regression model is then trained to predict sentiment.

# In[9]:


veczr =  CountVectorizer(ngram_range=(1,3), tokenizer=tokenize, max_features=800000)
trn_term_doc = veczr.fit_transform(trn)
val_term_doc = veczr.transform(val)


# In[10]:


trn_term_doc.shape


# In[11]:


vocab = veczr.get_feature_names()


# In[12]:


vocab[200000:200005]


# In[13]:


y=trn_y
x=trn_term_doc.sign()
val_x = val_term_doc.sign()


# In[16]:


r = np.log(pr(1) / pr(0))
b = np.log((y==1).mean() / (y==0).mean())


# Here we fit regularized logistic regression where the features are the trigrams.

# In[42]:


m = LogisticRegression(C=0.1, dual=True)
m.fit(x, y);

preds = m.predict(val_x)
(preds.T==val_y).mean()


# Here is the $\text{log-count ratio}$ `r`.  

# In[43]:


r.shape, r


# In[44]:


np.exp(r)


# Here we fit regularized logistic regression where the features are the trigrams' log-count ratios.

# In[45]:


x_nb = x.multiply(r)
m = LogisticRegression(dual=True, C=0.1)
m.fit(x_nb, y);

val_x_nb = val_x.multiply(r)
preds = m.predict(val_x_nb)
(preds.T==val_y).mean()


# ## fastai NBSVM++

# In[17]:


sl=2000


# In[18]:


# Here is how we get a model from a bag of words
md = TextClassifierData.from_bow(trn_term_doc, trn_y, val_term_doc, val_y, sl)


# In[19]:


learner = md.dotprod_nb_learner()
learner.fit(0.02, 1, wds=1e-6, cycle_len=1)


# In[159]:


learner.fit(0.02, 2, wds=1e-6, cycle_len=1)


# In[160]:


learner.fit(0.02, 2, wds=1e-6, cycle_len=1)


# ## References

# * Baselines and Bigrams: Simple, Good Sentiment and Topic Classification. Sida Wang and Christopher D. Manning [pdf](https://www.aclweb.org/anthology/P12-2018)
