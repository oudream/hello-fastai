
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

from fastai.nlp import *
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from torchtext import vocab, data, datasets
import pandas as pd


# In[2]:


sl=1000
vocab_size=200000


# In[3]:


PATH='data/arxiv/arxiv.csv'

# You can download a similar to Jeremy's original arxiv.csv here: https://drive.google.com/file/d/0B34BjUTAgwm6SzdPWDAtVG1vWVU/. It comes from this article https://hackernoon.com/building-brundage-bot-10252facf3d1 and github https://github.com/amauboussin/arxiv-twitterbot, just rename it to arxiv.csv

df = pd.read_csv(PATH)
df.head()


# In[25]:


df['txt'] = df.category + ' ' + df.title + '\n' + df.summary


# In[28]:


print(df.iloc[0].txt)


# In[27]:


n=len(df); n


# In[76]:


val_idx = get_cv_idxs(n, val_pct=0.1)
((val,trn),(val_y,trn_y)) = split_by_idx(val_idx, df.txt.values, df.tweeted.values)


# ## Ngram logistic regression

# In[77]:


veczr =  CountVectorizer(ngram_range=(1,3), tokenizer=tokenize)
trn_term_doc = veczr.fit_transform(trn)
val_term_doc = veczr.transform(val)
trn_term_doc.shape, trn_term_doc.sum()


# In[78]:


y=trn_y
x=trn_term_doc.sign()
val_x = val_term_doc.sign()


# In[79]:


p = x[np.argwhere(y!=0)[:,0]].sum(0)+1
q = x[np.argwhere(y==0)[:,0]].sum(0)+1
r = np.log((p/p.sum())/(q/q.sum()))
b = np.log(len(p)/len(q))


# In[80]:


pre_preds = val_term_doc @ r.T + b
preds = pre_preds.T>0
(preds==val_y).mean()


# In[81]:


m = LogisticRegression(C=0.1, fit_intercept=False)
m.fit(x, y);

preds = m.predict(val_x)
(preds.T==val_y).mean()


# In[82]:


probs = m.predict_proba(val_x)[:,1]


# In[83]:


from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

precision, recall, _ = precision_recall_curve(val_y, probs)
average_precision = average_precision_score(val_y, probs)

plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curve: AUC={0:0.2f}'.format(average_precision));


# In[84]:


recall[precision>=0.6][0]


# In[97]:


df_val = df.iloc[sorted(val_idx)]


# In[90]:


incorrect_yes = np.where((preds != val_y) & (val_y == 0))[0]
most_incorrect_yes = np.argsort(-probs[incorrect_yes])
txts = df_val.iloc[incorrect_yes[most_incorrect_yes[:10]]]
txts[["link", "title", "summary"]]


# In[106]:


' '.join(txts.link.values)


# In[115]:


incorrect_no = np.where((preds != val_y) & (val_y == 1))[0]
most_incorrect_no = np.argsort(probs[incorrect_no])
txts = df_val.iloc[incorrect_no[most_incorrect_no[:10]]]


# In[117]:


txts[["link", "title", "summary"]]


# In[119]:


' '.join(txts.link.values)


# In[ ]:


to_review = np.where((preds > 0.8) & (val_y == 0))[0]
to_review_idx = np.argsort(-probs[to_review])
txts = df_val.iloc[to_review[to_review_idx]]


# In[130]:


txt_html = ('<li><a href="http://' + txts.link + '">' + txts.title.str.replace('\n',' ') + '</a>: ' 
    + txts.summary.str.replace('\n',' ') + '</li>').values


# In[136]:


full_html = (f"""<!DOCTYPE html>
<html>
<head><title>Brundage Bot Backfill</title></head>
<body>
<ul>
{os.linesep.join(txt_html)}
</ul>
</body>
</html>""")


# ## Learner

# In[137]:


veczr = CountVectorizer(ngram_range=(1,3), tokenizer=tokenize, max_features=vocab_size)

trn_term_doc = veczr.fit_transform(trn)
val_term_doc = veczr.transform(val)


# In[138]:


trn_term_doc.shape, trn_term_doc.sum()


# In[139]:


md = TextClassifierData.from_bow(trn_term_doc, trn_y, val_term_doc, val_y, sl)


# In[141]:


learner = md.dotprod_nb_learner(r_adj=20)


# In[142]:


learner.fit(0.02, 4, wds=1e-6, cycle_len=1)


# In[140]:


from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

def prec_at_6(preds,targs):
    precision, recall, _ = precision_recall_curve(targs[:,1], preds[:,1])
    return recall[precision>=0.6][0]


# In[144]:


prec_at_6(*learner.predict_with_targs())

