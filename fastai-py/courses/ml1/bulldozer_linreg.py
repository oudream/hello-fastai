
# coding: utf-8

# # A Linear Model for Bulldozers

# In[4]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')

from fastai.imports import *
from fastai.structured import *
from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
from sklearn import metrics
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV


# In[6]:


set_plot_sizes(12,14,16)


# ## Load in our data from last lesson

# In[7]:


PATH = "data/bulldozers/"

df_raw = pd.read_feather('tmp/bulldozers-raw')


# In[10]:


df_raw['age'] = df_raw.saleYear-df_raw.YearMade


# In[11]:


df, y, nas, mapper = proc_df(df_raw, 'SalePrice', max_n_cat=10, do_scale=True)


# In[12]:


def split_vals(a,n): return a[:n], a[n:]
n_valid = 12000
n_trn = len(df)-n_valid
y_train, y_valid = split_vals(y, n_trn)
raw_train, raw_valid = split_vals(df_raw, n_trn)


# In[13]:


def rmse(x,y): return math.sqrt(((x-y)**2).mean())


# # Linear regression for Bulldozers

# ## Data scaling

# In[14]:


df.describe().transpose()


# In[15]:


X_train, X_valid = split_vals(df, n_trn)


# In[16]:


m = LinearRegression().fit(X_train, y_train)
m.score(X_valid, y_valid)


# In[17]:


m.score(X_train, y_train)


# In[18]:


preds = m.predict(X_valid)


# In[19]:


rmse(preds, y_valid)


# In[20]:


plt.scatter(preds, y_valid, alpha=0.1, s=2);


# ## Feature selection from RF

# In[204]:


keep_cols = list(np.load('tmp/keep_cols.npy'))
', '.join(keep_cols)


# In[205]:


df_sub = df_raw[keep_cols+['age', 'SalePrice']]


# In[206]:


df, y, nas, mapper = proc_df(df_sub, 'SalePrice', max_n_cat=10, do_scale=True)


# In[207]:


X_train, X_valid = split_vals(df, n_trn)


# In[208]:


m = LinearRegression().fit(X_train, y_train)
m.score(X_valid, y_valid)


# In[209]:


rmse(m.predict(X_valid), y_valid)


# In[210]:


from operator import itemgetter


# In[211]:


sorted(list(zip(X_valid.columns, m.coef_)), key=itemgetter(1))


# In[212]:


m = LassoCV().fit(X_train, y_train)
m.score(X_valid, y_valid)


# In[213]:


rmse(m.predict(X_valid), y_valid)


# In[214]:


m.alpha_


# In[215]:


coefs = sorted(list(zip(X_valid.columns, m.coef_)), key=itemgetter(1))
coefs


# In[216]:


skip = [n for n,c in coefs if abs(c)<0.01]


# In[217]:


df.drop(skip, axis=1, inplace=True)

# for n,c in df.items():
#     if '_' not in n: df[n+'2'] = df[n]**2


# In[218]:


X_train, X_valid = split_vals(df, n_trn)


# In[219]:


m = LassoCV().fit(X_train, y_train)
m.score(X_valid, y_valid)


# In[220]:


rmse(m.predict(X_valid), y_valid)


# In[221]:


coefs = sorted(list(zip(X_valid.columns, m.coef_)), key=itemgetter(1))
coefs


# In[222]:


np.savez(f'{PATH}tmp/regr_resid', m.predict(X_train), m.predict(X_valid))

