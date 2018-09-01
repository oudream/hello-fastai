
# coding: utf-8

# # Random Forest Model interpretation

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')

from fastai.imports import *
from fastai.structured import *
from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
from sklearn import metrics


# In[3]:


set_plot_sizes(12,14,16)


# ## Load in our data from last lesson

# In[4]:


PATH = "data/bulldozers/"

df_raw = pd.read_feather('tmp/bulldozers-raw')
df_trn, y_trn, nas = proc_df(df_raw, 'SalePrice')


# In[5]:


def split_vals(a,n): return a[:n], a[n:]
n_valid = 12000
n_trn = len(df_trn)-n_valid
X_train, X_valid = split_vals(df_trn, n_trn)
y_train, y_valid = split_vals(y_trn, n_trn)
raw_train, raw_valid = split_vals(df_raw, n_trn)


# In[6]:


def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)


# In[7]:


df_raw


# # Confidence based on tree variance

# For model interpretation, there's no need to use the full dataset on each tree - using a subset will be both faster, and also provide better interpretability (since an overfit model will not provide much variance across trees).

# In[7]:


set_rf_samples(50000)


# In[8]:


m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# We saw how the model averages predictions across the trees to get an estimate - but how can we know the confidence of the estimate? One simple way is to use the standard deviation of predictions, instead of just the mean. This tells us the *relative* confidence of predictions - that is, for rows where the trees give very different results, you would want to be more cautious of using those results, compared to cases where they are more consistent. Using the same example as in the last lesson when we looked at bagging:

# In[19]:


get_ipython().run_line_magic('time', 'preds = np.stack([t.predict(X_valid) for t in m.estimators_])')
np.mean(preds[:,0]), np.std(preds[:,0])


# When we use python to loop through trees like this, we're calculating each in series, which is slow! We can use parallel processing to speed things up:

# In[20]:


def get_preds(t): return t.predict(X_valid)
get_ipython().run_line_magic('time', 'preds = np.stack(parallel_trees(m, get_preds))')
np.mean(preds[:,0]), np.std(preds[:,0])


# We can see that different trees are giving different estimates this this auction. In order to see how prediction confidence varies, we can add this into our dataset.

# In[21]:


x = raw_valid.copy()
x['pred_std'] = np.std(preds, axis=0)
x['pred'] = np.mean(preds, axis=0)
x.Enclosure.value_counts().plot.barh();


# In[22]:


flds = ['Enclosure', 'SalePrice', 'pred', 'pred_std']
enc_summ = x[flds].groupby('Enclosure', as_index=False).mean()
enc_summ


# In[19]:


enc_summ = enc_summ[~pd.isnull(enc_summ.SalePrice)]
enc_summ.plot('Enclosure', 'SalePrice', 'barh', xlim=(0,11));


# In[20]:


enc_summ.plot('Enclosure', 'pred', 'barh', xerr='pred_std', alpha=0.6, xlim=(0,11));


# *Question*: Why are the predictions nearly exactly right, but the error bars are quite wide?

# In[21]:


raw_valid.ProductSize.value_counts().plot.barh();


# In[22]:


flds = ['ProductSize', 'SalePrice', 'pred', 'pred_std']
summ = x[flds].groupby(flds[0]).mean()
summ


# In[23]:


(summ.pred_std/summ.pred).sort_values(ascending=False)


# # Feature importance

# It's not normally enough to just to know that a model can make accurate predictions - we also want to know *how* it's making predictions. The most important way to see this is with *feature importance*.

# In[23]:


fi = rf_feat_importance(m, df_trn); fi[:10]


# In[24]:


fi.plot('cols', 'imp', figsize=(10,6), legend=False);


# In[25]:


def plot_fi(fi): return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)


# In[26]:


plot_fi(fi[:30]);


# In[27]:


to_keep = fi[fi.imp>0.005].cols; len(to_keep)


# In[28]:


df_keep = df_trn[to_keep].copy()
X_train, X_valid = split_vals(df_keep, n_trn)


# In[29]:


m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5,
                          n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[30]:


fi = rf_feat_importance(m, df_keep)
plot_fi(fi);


# ## One-hot encoding

# proc_df's optional *max_n_cat* argument will turn some categorical variables into new columns.
# 
# For example, the column **ProductSize** which has 6 categories:
# 
# * Large
# * Large / Medium
# * Medium
# * Compact
# * Small
# * Mini
# 
# gets turned into 6 new columns:
# 
# * ProductSize_Large
# * ProductSize_Large / Medium
# * ProductSize_Medium
# * ProductSize_Compact
# * ProductSize_Small
# * ProductSize_Mini
# 
# and the column **ProductSize** gets removed.
# 
# It will only happen to columns whose number of categories is no bigger than the value of the *max_n_cat* argument.
# 
# Now some of these new columns may prove to have more important features than in the earlier situation, where all categories were in one column.

# In[31]:


df_trn2, y_trn, nas = proc_df(df_raw, 'SalePrice', max_n_cat=7)
X_train, X_valid = split_vals(df_trn2, n_trn)

m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.6, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[32]:


fi = rf_feat_importance(m, df_trn2)
plot_fi(fi[:25]);


# # Removing redundant features

# One thing that makes this harder to interpret is that there seem to be some variables with very similar meanings. Let's try to remove redundent features.

# In[33]:


from scipy.cluster import hierarchy as hc


# In[34]:


corr = np.round(scipy.stats.spearmanr(df_keep).correlation, 4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')
fig = plt.figure(figsize=(16,10))
dendrogram = hc.dendrogram(z, labels=df_keep.columns, orientation='left', leaf_font_size=16)
plt.show()


# Let's try removing some of these related features to see if the model can be simplified without impacting the accuracy.

# In[35]:


def get_oob(df):
    m = RandomForestRegressor(n_estimators=30, min_samples_leaf=5, max_features=0.6, n_jobs=-1, oob_score=True)
    x, _ = split_vals(df, n_trn)
    m.fit(x, y_train)
    return m.oob_score_


# Here's our baseline.

# In[36]:


get_oob(df_keep)


# Now we try removing each variable one at a time.

# In[46]:


for c in ('saleYear', 'saleElapsed', 'fiModelDesc', 'fiBaseModel', 'Grouser_Tracks', 'Coupler_System'):
    print(c, get_oob(df_keep.drop(c, axis=1)))


# It looks like we can try one from each group for removal. Let's see what that does.

# In[37]:


to_drop = ['saleYear', 'fiBaseModel', 'Grouser_Tracks']
get_oob(df_keep.drop(to_drop, axis=1))


# Looking good! Let's use this dataframe from here. We'll save the list of columns so we can reuse it later.

# In[38]:


df_keep.drop(to_drop, axis=1, inplace=True)
X_train, X_valid = split_vals(df_keep, n_trn)


# In[39]:


np.save('tmp/keep_cols.npy', np.array(df_keep.columns))


# In[9]:


keep_cols = np.load('tmp/keep_cols.npy')
df_keep = df_trn[keep_cols]


# And let's see how this model looks on the full dataset.

# In[40]:


reset_rf_samples()


# In[41]:


m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# # Partial dependence

# In[10]:


from pdpbox import pdp
from plotnine import *


# In[11]:


set_rf_samples(50000)


# This next analysis will be a little easier if we use the 1-hot encoded categorical variables, so let's load them up again.

# In[46]:


df_trn2, y_trn, nas = proc_df(df_raw, 'SalePrice', max_n_cat=7)
X_train, X_valid = split_vals(df_trn2, n_trn)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.6, n_jobs=-1)
m.fit(X_train, y_train);


# In[47]:


plot_fi(rf_feat_importance(m, df_trn2)[:10]);


# In[48]:


df_raw.plot('YearMade', 'saleElapsed', 'scatter', alpha=0.01, figsize=(10,8));


# In[49]:


x_all = get_sample(df_raw[df_raw.YearMade>1930], 500)


# In[46]:


ggplot(x_all, aes('YearMade', 'SalePrice'))+stat_smooth(se=True, method='loess')


# In[47]:


x = get_sample(X_train[X_train.YearMade>1930], 500)


# In[48]:


def plot_pdp(feat, clusters=None, feat_name=None):
    feat_name = feat_name or feat
    p = pdp.pdp_isolate(m, x, feat)
    return pdp.pdp_plot(p, feat_name, plot_lines=True, 
                        cluster=clusters is not None, n_cluster_centers=clusters)


# In[49]:


plot_pdp('YearMade')


# In[50]:


plot_pdp('YearMade', clusters=5)


# In[51]:


feats = ['saleElapsed', 'YearMade']
p = pdp.pdp_interact(m, x, feats)
pdp.pdp_interact_plot(p, feats)


# In[52]:


plot_pdp(['Enclosure_EROPS w AC', 'Enclosure_EROPS', 'Enclosure_OROPS'], 5, 'Enclosure')


# In[50]:


df_raw.YearMade[df_raw.YearMade<1950] = 1950
df_keep['age'] = df_raw['age'] = df_raw.saleYear-df_raw.YearMade


# In[51]:


X_train, X_valid = split_vals(df_keep, n_trn)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.6, n_jobs=-1)
m.fit(X_train, y_train)
plot_fi(rf_feat_importance(m, df_keep));


# # Tree interpreter

# In[12]:


from treeinterpreter import treeinterpreter as ti


# In[13]:


df_train, df_valid = split_vals(df_raw[df_keep.columns], n_trn)


# In[14]:


row = X_valid.values[None,0]; row


# In[15]:


prediction, bias, contributions = ti.predict(m, row)


# In[16]:


prediction[0], bias[0]


# In[58]:


idxs = np.argsort(contributions[0])


# In[61]:


[o for o in zip(df_keep.columns[idxs], df_valid.iloc[0][idxs], contributions[0][idxs])]


# In[61]:


contributions[0].sum()


# # Extrapolation

# In[58]:


df_ext = df_keep.copy()
df_ext['is_valid'] = 1
df_ext.is_valid[:n_trn] = 0
x, y, nas = proc_df(df_ext, 'is_valid')


# In[59]:


m = RandomForestClassifier(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(x, y);
m.oob_score_


# In[60]:


fi = rf_feat_importance(m, x); fi[:10]


# In[61]:


feats=['SalesID', 'saleElapsed', 'MachineID']


# In[62]:


(X_train[feats]/1000).describe()


# In[63]:


(X_valid[feats]/1000).describe()


# In[64]:


x.drop(feats, axis=1, inplace=True)


# In[65]:


m = RandomForestClassifier(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(x, y);
m.oob_score_


# In[66]:


fi = rf_feat_importance(m, x); fi[:10]


# In[97]:


set_rf_samples(50000)


# In[67]:


feats=['SalesID', 'saleElapsed', 'MachineID', 'age', 'YearMade', 'saleDayofyear']


# In[68]:


X_train, X_valid = split_vals(df_keep, n_trn)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[69]:


for f in feats:
    df_subs = df_keep.drop(f, axis=1)
    X_train, X_valid = split_vals(df_subs, n_trn)
    m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
    m.fit(X_train, y_train)
    print(f)
    print_score(m)


# In[77]:


reset_rf_samples()


# In[78]:


df_subs = df_keep.drop(['SalesID', 'MachineID', 'saleDayofyear'], axis=1)
X_train, X_valid = split_vals(df_subs, n_trn)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[79]:


plot_fi(rf_feat_importance(m, X_train));


# In[80]:


np.save('tmp/subs_cols.npy', np.array(df_subs.columns))


# # Our final model!

# In[81]:


m = RandomForestRegressor(n_estimators=160, max_features=0.5, n_jobs=-1, oob_score=True)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)

