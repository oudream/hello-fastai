
# coding: utf-8

# This notebook contains an implementation of the third place result in the Rossman Kaggle competition as detailed in Guo/Berkhahn's [Entity Embeddings of Categorical Variables](https://arxiv.org/abs/1604.06737).

# The motivation behind exploring this architecture is it's relevance to real-world application. Much of our focus has been computer-vision and NLP tasks, which largely deals with unstructured data.
# 
# However, most of the data informing KPI's in industry are structured, time-series data. Here we explore the end-to-end process of using neural networks with practical structured data problems.

# In[1]:


get_ipython().magic(u'matplotlib inline')


# In[2]:


import math, keras, datetime, pandas as pd, numpy as np, keras.backend as K
import matplotlib.pyplot as plt, xgboost, operator, random, pickle


# In[3]:


from utils2 import *


# In[4]:


np.set_printoptions(threshold=50, edgeitems=20)


# In[5]:


limit_mem()


# In[8]:


from isoweek import Week
from pandas_summary import DataFrameSummary


# In[9]:


get_ipython().magic(u'cd /data/datasets/rossman/')


# ## Create datasets

# In addition to the provided data, we will be using external datasets put together by participants in the Kaggle competition. You can download all of them [here](http://files.fast.ai/part2/lesson14/rossmann.tgz).
# 
# For completeness, the implementation used to put them together is included below.

# In[11]:


def concat_csvs(dirname):
    os.chdir(dirname)
    filenames=glob.glob("*.csv")

    wrote_header = False
    with open("../"+dirname+".csv","w") as outputfile:
        for filename in filenames:
            name = filename.split(".")[0]
            with open(filename) as f:
                line = f.readline()
                if not wrote_header:
                    wrote_header = True
                    outputfile.write("file,"+line)
                for line in f:
                     outputfile.write(name + "," + line)
                outputfile.write("\n")

    os.chdir("..")


# In[13]:


# concat_csvs('googletrend')
# concat_csvs('weather')


# Feature Space:
# * train: Training set provided by competition
# * store: List of stores
# * store_states: mapping of store to the German state they are in
# * List of German state names
# * googletrend: trend of certain google keywords over time, found by users to correlate well w/ given data
# * weather: weather
# * test: testing set

# In[42]:


table_names = ['train', 'store', 'store_states', 'state_names', 
               'googletrend', 'weather', 'test']


# We'll be using the popular data manipulation framework pandas.
# 
# Among other things, pandas allows you to manipulate tables/data frames in python as one would in a database.

# We're going to go ahead and load all of our csv's as dataframes into a list `tables`.

# In[43]:


tables = [pd.read_csv(fname+'.csv', low_memory=False) for fname in table_names]


# In[16]:


from IPython.display import HTML


# We can use `head()` to get a quick look at the contents of each table:
# * train: Contains store information on a daily basis, tracks things like sales, customers, whether that day was a holdiay, etc.
# * store: general info about the store including competition, etc.
# * store_states: maps store to state it is in
# * state_names: Maps state abbreviations to names
# * googletrend: trend data for particular week/state
# * weather: weather conditions for each state
# * test: Same as training table, w/o sales and customers
# 

# In[17]:


for t in tables: display(t.head())


# This is very representative of a typical industry dataset.

# The following returns summarized aggregate information to each table accross each field.

# In[41]:


for t in tables: display(DataFrameSummary(t).summary())


# ## Data Cleaning / Feature Engineering

# As a structured data problem, we necessarily have to go through all the cleaning and feature engineering, even though we're using a neural network.

# In[44]:


train, store, store_states, state_names, googletrend, weather, test = tables


# In[45]:


len(train),len(test)


# Turn state Holidays to Bool

# In[46]:


train.StateHoliday = train.StateHoliday!='0'
test.StateHoliday = test.StateHoliday!='0'


# Define function for joining tables on specific fields.
# 
# By default, we'll be doing a left outer join of `right` on the `left` argument using the given fields for each table.
# 
# Pandas does joins using the `merge` method. The `suffixes` argument describes the naming convention for duplicate fields. We've elected to leave the duplicate field names on the left untouched, and append a "_y" to those on the right.

# In[47]:


def join_df(left, right, left_on, right_on=None):
    if right_on is None: right_on = left_on
    return left.merge(right, how='left', left_on=left_on, right_on=right_on, 
                      suffixes=("", "_y"))


# Join weather/state names.

# In[48]:


weather = join_df(weather, state_names, "file", "StateName")


# In pandas you can add new columns to a dataframe by simply defining it. We'll do this for googletrends by extracting dates and state names from the given data and adding those columns.
# 
# We're also going to replace all instances of state name 'NI' with the usage in the rest of the table, 'HB,NI'. This is a good opportunity to highlight pandas indexing. We can use `.ix[rows, cols]` to select a list of rows and a list of columns from the dataframe. In this case, we're selecting rows w/ statename 'NI' by using a boolean list `googletrend.State=='NI'` and selecting "State".

# In[49]:


googletrend['Date'] = googletrend.week.str.split(' - ', expand=True)[0]
googletrend['State'] = googletrend.file.str.split('_', expand=True)[2]
googletrend.loc[googletrend.State=='NI', "State"] = 'HB,NI'


# The following extracts particular date fields from a complete datetime for the purpose of constructing categoricals.
# 
# You should always consider this feature extraction step when working with date-time. Without expanding your date-time into these additional fields, you can't capture any trend/cyclical behavior as a function of time at any of these granularities.

# In[50]:


def add_datepart(df):
    df.Date = pd.to_datetime(df.Date)
    df["Year"] = df.Date.dt.year
    df["Month"] = df.Date.dt.month
    df["Week"] = df.Date.dt.week
    df["Day"] = df.Date.dt.day


# We'll add to every table w/ a date field.

# In[51]:


add_datepart(weather)
add_datepart(googletrend)
add_datepart(train)
add_datepart(test)


# In[52]:


trend_de = googletrend[googletrend.file == 'Rossmann_DE']


# Now we can outer join all of our data into a single dataframe.
# 
# Recall that in outer joins everytime a value in the joining field on the left table does not have a corresponding value on the right table, the corresponding row in the new table has Null values for all right table fields.
# 
# One way to check that all records are consistent and complete is to check for Null values post-join, as we do here.
# 
# *Aside*: Why note just do an inner join?
# If you are assuming that all records are complete and match on the field you desire, an inner join will do the same thing as an outer join. However, in the event you are wrong or a mistake is made, an outer join followed by a null-check will catch it. (Comparing before/after # of rows for inner join is equivalent, but requires keeping track of before/after row #'s. Outer join is easier.)

# In[53]:


store = join_df(store, store_states, "Store")
len(store[store.State.isnull()])


# In[54]:


joined = join_df(train, store, "Store")
len(joined[joined.StoreType.isnull()])


# In[55]:


joined = join_df(joined, googletrend, ["State","Year", "Week"])
len(joined[joined.trend.isnull()])


# In[56]:


joined = joined.merge(trend_de, 'left', ["Year", "Week"], suffixes=('', '_DE'))
len(joined[joined.trend_DE.isnull()])


# In[57]:


joined = join_df(joined, weather, ["State","Date"])
len(joined[joined.Mean_TemperatureC.isnull()])


# In[58]:


joined_test = test.merge(store, how='left', left_on='Store', right_index=True)
len(joined_test[joined_test.StoreType.isnull()])


# Next we'll fill in missing values to avoid complications w/ na's.

# In[59]:


joined.CompetitionOpenSinceYear = joined.CompetitionOpenSinceYear.fillna(1900).astype(np.int32)
joined.CompetitionOpenSinceMonth = joined.CompetitionOpenSinceMonth.fillna(1).astype(np.int32)
joined.Promo2SinceYear = joined.Promo2SinceYear.fillna(1900).astype(np.int32)
joined.Promo2SinceWeek = joined.Promo2SinceWeek.fillna(1).astype(np.int32)


# Next we'll extract features "CompetitionOpenSince" and "CompetitionDaysOpen". Note the use of `apply()` in mapping a function across dataframe values.

# In[60]:


joined["CompetitionOpenSince"] = pd.to_datetime(joined.apply(lambda x: datetime.datetime(
    x.CompetitionOpenSinceYear, x.CompetitionOpenSinceMonth, 15), axis=1).astype(pd.datetime))
joined["CompetitionDaysOpen"] = joined.Date.subtract(joined["CompetitionOpenSince"]).dt.days


# We'll replace some erroneous / outlying data.

# In[63]:


joined.loc[joined.CompetitionDaysOpen<0, "CompetitionDaysOpen"] = 0
joined.loc[joined.CompetitionOpenSinceYear<1990, "CompetitionDaysOpen"] = 0


# Added "CompetitionMonthsOpen" field, limit the maximum to 2 years to limit number of unique embeddings.

# In[64]:


joined["CompetitionMonthsOpen"] = joined["CompetitionDaysOpen"]//30
joined.loc[joined.CompetitionMonthsOpen>24, "CompetitionMonthsOpen"] = 24
joined.CompetitionMonthsOpen.unique()


# Same process for Promo dates.

# In[65]:


joined["Promo2Since"] = pd.to_datetime(joined.apply(lambda x: Week(
    x.Promo2SinceYear, x.Promo2SinceWeek).monday(), axis=1).astype(pd.datetime))
joined["Promo2Days"] = joined.Date.subtract(joined["Promo2Since"]).dt.days


# In[66]:


joined.loc[joined.Promo2Days<0, "Promo2Days"] = 0
joined.loc[joined.Promo2SinceYear<1990, "Promo2Days"] = 0


# In[67]:


joined["Promo2Weeks"] = joined["Promo2Days"]//7
joined.loc[joined.Promo2Weeks<0, "Promo2Weeks"] = 0
joined.loc[joined.Promo2Weeks>25, "Promo2Weeks"] = 25
joined.Promo2Weeks.unique()


# ## Durations

# It is common when working with time series data to extract data that explains relationships across rows as opposed to columns, e.g.:
# * Running averages
# * Time until next event
# * Time since last event
# 
# This is often difficult to do with most table manipulation frameworks, since they are designed to work with relationships across columns. As such, we've created a class to handle this type of data.

# In[72]:


columns = ["Date", "Store", "Promo", "StateHoliday", "SchoolHoliday"]


# We've defined a class `elapsed` for cumulative counting across a sorted dataframe.
# 
# Given a particular field `fld` to monitor, this object will start tracking time since the last occurrence of that field. When the field is seen again, the counter is set to zero.
# 
# Upon initialization, this will result in datetime na's until the field is encountered. This is reset every time a new store is seen.
# 
# We'll see how to use this shortly.

# In[73]:


class elapsed(object):
    def __init__(self, fld):
        self.fld = fld
        self.last = pd.to_datetime(np.nan)
        self.last_store = 0
        
    def get(self, row):
        if row.Store != self.last_store:
            self.last = pd.to_datetime(np.nan)
            self.last_store = row.Store
        if (row[self.fld]): self.last = row.Date
        return row.Date-self.last


# In[74]:


df = train[columns]


# And a function for applying said class across dataframe rows and adding values to a new column.

# In[75]:


def add_elapsed(fld, prefix):
    sh_el = elapsed(fld)
    df[prefix+fld] = df.apply(sh_el.get, axis=1)


# Let's walk through an example.
# 
# Say we're looking at School Holiday. We'll first sort by Store, then Date, and then call `add_elapsed('SchoolHoliday', 'After')`:
# This will generate an instance of the `elapsed` class for School Holiday:
# * Instance applied to every row of the dataframe in order of store and date
# * Will add to the dataframe the days since seeing a School Holiday
# * If we sort in the other direction, this will count the days until another promotion.

# In[76]:


fld = 'SchoolHoliday'
df = df.sort_values(['Store', 'Date'])
add_elapsed(fld, 'After')
df = df.sort_values(['Store', 'Date'], ascending=[True, False])
add_elapsed(fld, 'Before')


# We'll do this for two more fields.

# In[77]:


fld = 'StateHoliday'
df = df.sort_values(['Store', 'Date'])
add_elapsed(fld, 'After')
df = df.sort_values(['Store', 'Date'], ascending=[True, False])
add_elapsed(fld, 'Before')


# In[78]:


fld = 'Promo'
df = df.sort_values(['Store', 'Date'])
add_elapsed(fld, 'After')
df = df.sort_values(['Store', 'Date'], ascending=[True, False])
add_elapsed(fld, 'Before')


# We're going to set the active index to Date.

# In[79]:


df = df.set_index("Date")


# Then set null values from elapsed field calculations to 0.

# In[80]:


columns = ['SchoolHoliday', 'StateHoliday', 'Promo']


# In[81]:


for o in ['Before', 'After']:
    for p in columns:
        a = o+p
        df[a] = df[a].fillna(pd.Timedelta(0)).dt.days


# Next we'll demonstrate window functions in pandas to calculate rolling quantities.
# 
# Here we're sorting by date (`sort_index()`) and counting the number of events of interest (`sum()`) defined in `columns` in the following week (`rolling()`), grouped by Store (`groupby()`). We do the same in the opposite direction.

# In[83]:


bwd = df[['Store']+columns].sort_index().groupby("Store").rolling(7, min_periods=1).sum()


# In[84]:


fwd = df[['Store']+columns].sort_index(ascending=False
                                      ).groupby("Store").rolling(7, min_periods=1).sum()


# Next we want to drop the Store indices grouped together in the window function.
# 
# Often in pandas, there is an option to do this in place. This is time and memory efficient when working with large datasets.

# In[85]:


bwd.drop('Store',1,inplace=True)
bwd.reset_index(inplace=True)


# In[90]:


fwd.drop('Store',1,inplace=True)
fwd.reset_index(inplace=True)


# In[91]:


df.reset_index(inplace=True)


# Now we'll merge these values onto the df.

# In[92]:


df = df.merge(bwd, 'left', ['Date', 'Store'], suffixes=['', '_bw'])
df = df.merge(fwd, 'left', ['Date', 'Store'], suffixes=['', '_fw'])


# In[94]:


df.drop(columns,1,inplace=True)


# In[100]:


df.head()


# It's usually a good idea to back up large tables of extracted / wrangled features before you join them onto another one, that way you can go back to it easily if you need to make changes to it.

# In[96]:


df.to_csv('df.csv')


# In[97]:


df = pd.read_csv('df.csv', index_col=0)


# In[98]:


df["Date"] = pd.to_datetime(df.Date)


# In[99]:


df.columns


# In[101]:


joined = join_df(joined, df, ['Store', 'Date'])


# We'll back this up as well.

# In[102]:


joined.to_csv('joined.csv')


# We now have our final set of engineered features.

# In[111]:


joined = pd.read_csv('joined.csv', index_col=0)
joined["Date"] = pd.to_datetime(joined.Date)
joined.columns


# While these steps were explicitly outlined in the paper, these are all fairly typical feature engineering steps for dealing with time series data and are practical in any similar setting.

# ## Create features

# Now that we've engineered all our features, we need to convert to input compatible with a neural network.
# 
# This includes converting categorical variables into contiguous integers or one-hot encodings, normalizing continuous features to standard normal, etc...

# In[104]:


from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder, Imputer, StandardScaler


# This dictionary maps categories to embedding dimensionality. In generally, categories we might expect to be conceptually more complex have larger dimension.

# In[105]:


cat_var_dict = {'Store': 50, 'DayOfWeek': 6, 'Year': 2, 'Month': 6,
'Day': 10, 'StateHoliday': 3, 'CompetitionMonthsOpen': 2,
'Promo2Weeks': 1, 'StoreType': 2, 'Assortment': 3, 'PromoInterval': 3,
'CompetitionOpenSinceYear': 4, 'Promo2SinceYear': 4, 'State': 6,
'Week': 2, 'Events': 4, 'Promo_fw': 1,
'Promo_bw': 1, 'StateHoliday_fw': 1,
'StateHoliday_bw': 1, 'SchoolHoliday_fw': 1,
'SchoolHoliday_bw': 1}


# Name categorical variables

# In[106]:


cat_vars = [o[0] for o in 
            sorted(cat_var_dict.items(), key=operator.itemgetter(1), reverse=True)]


# In[107]:


"""cat_vars = ['Store', 'DayOfWeek', 'Year', 'Month', 'Day', 'StateHoliday',
           'StoreType', 'Assortment', 'Week', 'Events', 'Promo2SinceYear',
            'CompetitionOpenSinceYear', 'PromoInterval', 'Promo', 'SchoolHoliday', 'State']"""


# Likewise for continuous

# In[108]:


# mean/max wind; min temp; cloud; min/mean humid; 
contin_vars = ['CompetitionDistance', 
   'Max_TemperatureC', 'Mean_TemperatureC', 'Min_TemperatureC',
   'Max_Humidity', 'Mean_Humidity', 'Min_Humidity', 'Max_Wind_SpeedKm_h', 
   'Mean_Wind_SpeedKm_h', 'CloudCover', 'trend', 'trend_DE',
   'AfterStateHoliday', 'BeforeStateHoliday', 'Promo', 'SchoolHoliday']


# In[109]:


"""contin_vars = ['CompetitionDistance', 'Max_TemperatureC', 'Mean_TemperatureC', 
   'Max_Humidity', 'trend', 'trend_DE', 'AfterStateHoliday', 'BeforeStateHoliday']"""


# Replace nulls w/ 0 for continuous, "" for categorical.

# In[112]:


for v in contin_vars: joined.loc[joined[v].isnull(), v] = 0
for v in cat_vars: joined.loc[joined[v].isnull(), v] = ""


# Here we create a list of tuples, each containing a variable and an instance of a transformer for that variable.
# 
# For categoricals, we use a label encoder that maps categories to continuous integers. For continuous variables, we standardize them.

# In[113]:


cat_maps = [(o, LabelEncoder()) for o in cat_vars]
contin_maps = [([o], StandardScaler()) for o in contin_vars]


# The same instances need to be used for the test set as well, so values are mapped/standardized appropriately.
# 
# DataFrame mapper will keep track of these variable-instance mappings.

# In[114]:


cat_mapper = DataFrameMapper(cat_maps)
cat_map_fit = cat_mapper.fit(joined)
cat_cols = len(cat_map_fit.features)
cat_cols


# In[115]:


contin_mapper = DataFrameMapper(contin_maps)
contin_map_fit = contin_mapper.fit(joined)
contin_cols = len(contin_map_fit.features)
contin_cols


# Example of first five rows of zeroth column being transformed appropriately.

# In[116]:


cat_map_fit.transform(joined)[0,:5], contin_map_fit.transform(joined)[0,:5]


# We can also pickle these mappings, which is great for portability!

# In[117]:


pickle.dump(contin_map_fit, open('contin_maps.pickle', 'wb'))
pickle.dump(cat_map_fit, open('cat_maps.pickle', 'wb'))


# In[119]:


[len(o[1].classes_) for o in cat_map_fit.features]


# ## Sample data

# Next, the authors removed all instances where the store had zero sale / was closed.

# In[121]:


joined_sales = joined[joined.Sales!=0]
n = len(joined_sales)


# We speculate that this may have cost them a higher standing in the competition. One reason this may be the case is that a little EDA reveals that there are often periods where stores are closed, typically for refurbishment. Before and after these periods, there are naturally spikes in sales that one might expect. Be ommitting this data from their training, the authors gave up the ability to leverage information about these periods to predict this otherwise volatile behavior.

# In[122]:


n


# We're going to run on a sample.

# In[123]:


samp_size = 100000
np.random.seed(42)
idxs = sorted(np.random.choice(n, samp_size, replace=False))


# In[124]:


joined_samp = joined_sales.iloc[idxs].set_index("Date")


# In[125]:


samp_size = n
joined_samp = joined_sales.set_index("Date")


# In time series data, cross-validation is not random. Instead, our holdout data is always the most recent data, as it would be in real application.

# We've taken the last 10% as our validation set.

# In[126]:


train_ratio = 0.9
train_size = int(samp_size * train_ratio)


# In[127]:


train_size


# In[128]:


joined_valid = joined_samp[train_size:]
joined_train = joined_samp[:train_size]
len(joined_valid), len(joined_train)


# Here's a preprocessor for our categoricals using our instance mapper.

# In[129]:


def cat_preproc(dat):
    return cat_map_fit.transform(dat).astype(np.int64)


# In[130]:


cat_map_train = cat_preproc(joined_train)
cat_map_valid = cat_preproc(joined_valid)


# Same for continuous.

# In[131]:


def contin_preproc(dat):
    return contin_map_fit.transform(dat).astype(np.float32)


# In[132]:


contin_map_train = contin_preproc(joined_train)
contin_map_valid = contin_preproc(joined_valid)


# Grab our targets.

# In[133]:


y_train_orig = joined_train.Sales
y_valid_orig = joined_valid.Sales


# Finally, the authors modified the target values by applying a logarithmic transformation and normalizing to unit scale by dividing by the maximum log value.
# 
# Log transformations are used on this type of data frequently to attain a nicer shape. 
# 
# Further by scaling to the unit interval we can now use a sigmoid output in our neural network. Then we can multiply by the maximum log value to get the original log value and transform back.

# In[134]:


max_log_y = np.max(np.log(joined_samp.Sales))
y_train = np.log(y_train_orig)/max_log_y
y_valid = np.log(y_valid_orig)/max_log_y


# Note: Some testing shows this doesn't make a big difference.

# In[1066]:


"""#y_train = np.log(y_train)
ymean=y_train_orig.mean()
ystd=y_train_orig.std()
y_train = (y_train_orig-ymean)/ystd
#y_valid = np.log(y_valid)
y_valid = (y_valid_orig-ymean)/ystd"""


# Root-mean-squared percent error is the metric Kaggle used for this competition.

# In[136]:


def rmspe(y_pred, targ = y_valid_orig):
    pct_var = (targ - y_pred)/targ
    return math.sqrt(np.square(pct_var).mean())


# These undo the target transformations.

# In[135]:


def log_max_inv(preds, mx = max_log_y):
    return np.exp(preds * mx)


# In[137]:


def normalize_inv(preds):
    return preds * ystd + ymean


# ## Create models

# Now we're ready to put together our models.

# Much of the following code has commented out portions / alternate implementations.

# In[739]:


"""
1 97s - loss: 0.0104 - val_loss: 0.0083
2 93s - loss: 0.0076 - val_loss: 0.0076
3 90s - loss: 0.0071 - val_loss: 0.0076
4 90s - loss: 0.0068 - val_loss: 0.0075
5 93s - loss: 0.0066 - val_loss: 0.0075
6 95s - loss: 0.0064 - val_loss: 0.0076
7 98s - loss: 0.0063 - val_loss: 0.0077
8 97s - loss: 0.0062 - val_loss: 0.0075
9 95s - loss: 0.0061 - val_loss: 0.0073
0 101s - loss: 0.0061 - val_loss: 0.0074
"""


# In[150]:


def split_cols(arr): return np.hsplit(arr,arr.shape[1])


# In[193]:


map_train = split_cols(cat_map_train) + [contin_map_train]
map_valid = split_cols(cat_map_valid) + [contin_map_valid]


# In[194]:


len(map_train)


# In[191]:


map_train = split_cols(cat_map_train) + split_cols(contin_map_train)
map_valid = split_cols(cat_map_valid) + split_cols(contin_map_valid)


# Helper function for getting categorical name and dim.

# In[173]:


def cat_map_info(feat): return feat[0], len(feat[1].classes_)


# In[174]:


cat_map_info(cat_map_fit.features[1])


# In[175]:


def my_init(scale):
    return lambda shape, name=None: initializations.uniform(shape, scale=scale, name=name)


# In[176]:


def emb_init(shape, name=None): 
    return initializations.uniform(shape, scale=2/(shape[1]+1), name=name)


# Helper function for constructing embeddings. Notice commented out codes, several different ways to compute embeddings at play.
# 
# Also, note we're flattening the embedding. Embeddings in Keras come out as an element of a sequence like we might use in a sequence of words; here we just want to concatenate them so we flatten the 1-vector sequence into a vector.

# In[177]:


def get_emb(feat):
    name, c = cat_map_info(feat)
    #c2 = cat_var_dict[name]
    c2 = (c+1)//2
    if c2>50: c2=50
    inp = Input((1,), dtype='int64', name=name+'_in')
    # , W_regularizer=l2(1e-6)
    u = Flatten(name=name+'_flt')(Embedding(c, c2, input_length=1, init=emb_init)(inp))
#     u = Flatten(name=name+'_flt')(Embedding(c, c2, input_length=1)(inp))
    return inp,u


# Helper function for continuous inputs.

# In[178]:


def get_contin(feat):
    name = feat[0][0]
    inp = Input((1,), name=name+'_in')
    return inp, Dense(1, name=name+'_d', init=my_init(1.))(inp)


# Let's build them.

# In[179]:


contin_inp = Input((contin_cols,), name='contin')
contin_out = Dense(contin_cols*10, activation='relu', name='contin_d')(contin_inp)
#contin_out = BatchNormalization()(contin_out)


# Now we can put them together. Given the inputs, continuous and categorical embeddings, we're going to concatenate all of them.
# 
# Next, we're going to pass through some dropout, then two dense layers w/ ReLU activations, then dropout again, then the sigmoid activation we mentioned earlier.

# In[180]:


embs = [get_emb(feat) for feat in cat_map_fit.features]
#conts = [get_contin(feat) for feat in contin_map_fit.features]
#contin_d = [d for inp,d in conts]
x = merge([emb for inp,emb in embs] + [contin_out], mode='concat')
#x = merge([emb for inp,emb in embs] + contin_d, mode='concat')

x = Dropout(0.02)(x)
x = Dense(1000, activation='relu', init='uniform')(x)
x = Dense(500, activation='relu', init='uniform')(x)
x = Dropout(0.2)(x)
x = Dense(1, activation='sigmoid')(x)

model = Model([inp for inp,emb in embs] + [contin_inp], x)
#model = Model([inp for inp,emb in embs] + [inp for inp,d in conts], x)
model.compile('adam', 'mean_absolute_error')
#model.compile(Adam(), 'mse')


# ### Start training

# In[ ]:


get_ipython().run_cell_magic(u'time', u'', u'hist = model.fit(map_train, y_train, batch_size=128, nb_epoch=25,\n                 verbose=0, validation_data=(map_valid, y_valid))')


# In[133]:


hist.history


# In[ ]:


plot_train(hist)


# In[ ]:


preds = np.squeeze(model.predict(map_valid, 1024))


# Result on validation data:  0.1678 (samp 150k, 0.75 trn)

# In[ ]:


log_max_inv(preds)


# In[1056]:


normalize_inv(preds)


# ## Using 3rd place data

# In[377]:


pkl_path = '/data/jhoward/github/entity-embedding-rossmann/'


# In[401]:


def load_pickle(fname): 
    return pickle.load(open(pkl_path+fname + '.pickle', 'rb'))


# In[402]:


[x_pkl_orig, y_pkl_orig] = load_pickle('feature_train_data')


# In[403]:


max_log_y_pkl = np.max(np.log(y_pkl_orig))
y_pkl = np.log(y_pkl_orig)/max_log_y_pkl


# In[404]:


pkl_vars = ['Open', 'Store', 'DayOfWeek', 'Promo', 'Year', 'Month', 'Day', 
     'StateHoliday', 'SchoolHoliday', 'CompetitionMonthsOpen', 'Promo2Weeks', 
    'Promo2Weeks_L', 'CompetitionDistance',
    'StoreType', 'Assortment', 'PromoInterval', 'CompetitionOpenSinceYear',
    'Promo2SinceYear', 'State', 'Week', 'Max_TemperatureC', 'Mean_TemperatureC', 
    'Min_TemperatureC', 'Max_Humidity', 'Mean_Humidity', 'Min_Humidity', 'Max_Wind_SpeedKm_h', 
    'Mean_Wind_SpeedKm_h', 'CloudCover','Events', 'Promo_fw', 'Promo_bw', 
    'StateHoliday_fw', 'StateHoliday_bw', 'AfterStateHoliday', 'BeforeStateHoliday', 
    'SchoolHoliday_fw', 'SchoolHoliday_bw', 'trend_DE', 'trend']


# In[405]:


x_pkl = np.array(x_pkl_orig)


# In[406]:


gt_enc = StandardScaler()
gt_enc.fit(x_pkl[:,-2:])


# In[407]:


x_pkl[:,-2:] = gt_enc.transform(x_pkl[:,-2:])


# In[408]:


x_pkl.shape


# In[386]:


x_pkl = x_pkl[idxs]
y_pkl = y_pkl[idxs]


# In[409]:


x_pkl_trn, x_pkl_val = x_pkl[:train_size], x_pkl[train_size:]
y_pkl_trn, y_pkl_val = y_pkl[:train_size], y_pkl[train_size:]


# In[355]:


x_pkl_trn.shape


# In[179]:


xgb_parms = {'learning_rate': 0.1, 'subsample': 0.6, 
             'colsample_bylevel': 0.6, 'silent': True, 'objective': 'reg:linear'}


# In[180]:


xdata_pkl = xgboost.DMatrix(x_pkl_trn, y_pkl_trn, feature_names=pkl_vars)


# In[181]:


xdata_val_pkl = xgboost.DMatrix(x_pkl_val, y_pkl_val, feature_names=pkl_vars)


# In[182]:


xgb_parms['seed'] = random.randint(0,1e9)
model_pkl = xgboost.train(xgb_parms, xdata_pkl)


# In[183]:


model_pkl.eval(xdata_val_pkl)


# In[ ]:


#0.117473


# In[184]:


importance = model_pkl.get_fscore()
importance = sorted(importance.items(), key=operator.itemgetter(1))

df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()

df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance');


# ### Neural net

# In[410]:


#np.savez_compressed('vars.npz', pkl_cats, pkl_contins)
#np.savez_compressed('deps.npz', y_pkl)


# In[411]:


pkl_cats = np.stack([x_pkl[:,pkl_vars.index(f)] for f in cat_vars], 1)
pkl_contins = np.stack([x_pkl[:,pkl_vars.index(f)] for f in contin_vars], 1)


# In[412]:


co_enc = StandardScaler().fit(pkl_contins)
pkl_contins = co_enc.transform(pkl_contins)


# In[413]:


pkl_contins_trn, pkl_contins_val = pkl_contins[:train_size], pkl_contins[train_size:]
pkl_cats_trn, pkl_cats_val = pkl_cats[:train_size], pkl_cats[train_size:]
y_pkl_trn, y_pkl_val = y_pkl[:train_size], y_pkl[train_size:]


# In[414]:


def get_emb_pkl(feat):
    name, c = cat_map_info(feat)
    c2 = (c+2)//3
    if c2>50: c2=50
    inp = Input((1,), dtype='int64', name=name+'_in')
    u = Flatten(name=name+'_flt')(Embedding(c, c2, input_length=1, init=emb_init)(inp))
    return inp,u


# In[415]:


n_pkl_contin = pkl_contins_trn.shape[1]
contin_inp = Input((n_pkl_contin,), name='contin')
contin_out = BatchNormalization()(contin_inp)


# In[416]:


map_train_pkl = split_cols(pkl_cats_trn) + [pkl_contins_trn]
map_valid_pkl = split_cols(pkl_cats_val) + [pkl_contins_val]


# In[417]:


def train_pkl(bs=128, ne=10):
    return model_pkl.fit(map_train_pkl, y_pkl_trn, batch_size=bs, nb_epoch=ne,
                 verbose=0, validation_data=(map_valid_pkl, y_pkl_val))


# In[418]:


def get_model_pkl(): 
    conts = [get_contin_pkl(feat) for feat in contin_map_fit.features]
    embs = [get_emb_pkl(feat) for feat in cat_map_fit.features]
    x = merge([emb for inp,emb in embs] + [contin_out], mode='concat')

    x = Dropout(0.02)(x)
    x = Dense(1000, activation='relu', init='uniform')(x)
    x = Dense(500, activation='relu', init='uniform')(x)
    x = Dense(1, activation='sigmoid')(x)

    model_pkl = Model([inp for inp,emb in embs] + [contin_inp], x)
    model_pkl.compile('adam', 'mean_absolute_error')
    #model.compile(Adam(), 'mse')
    return model_pkl


# In[458]:


model_pkl = get_model_pkl()


# In[459]:


train_pkl(128, 10).history['val_loss']


# In[460]:


K.set_value(model_pkl.optimizer.lr, 1e-4)
train_pkl(128, 5).history['val_loss']


# In[739]:


"""
1 97s - loss: 0.0104 - val_loss: 0.0083
2 93s - loss: 0.0076 - val_loss: 0.0076
3 90s - loss: 0.0071 - val_loss: 0.0076
4 90s - loss: 0.0068 - val_loss: 0.0075
5 93s - loss: 0.0066 - val_loss: 0.0075
6 95s - loss: 0.0064 - val_loss: 0.0076
7 98s - loss: 0.0063 - val_loss: 0.0077
8 97s - loss: 0.0062 - val_loss: 0.0075
9 95s - loss: 0.0061 - val_loss: 0.0073
0 101s - loss: 0.0061 - val_loss: 0.0074
"""


# In[116]:


plot_train(hist)


# In[1214]:


preds = np.squeeze(model_pkl.predict(map_valid_pkl, 1024))


# In[1222]:


y_orig_pkl_val = log_max_inv(y_pkl_val, max_log_y_pkl)


# In[1224]:


rmspe(log_max_inv(preds, max_log_y_pkl), y_orig_pkl_val)


# ## XGBoost

# Xgboost is extremely quick and easy to use. Aside from being a powerful predictive model, it gives us information about feature importance.

# In[52]:


X_train = np.concatenate([cat_map_train, contin_map_train], axis=1)


# In[53]:


X_valid = np.concatenate([cat_map_valid, contin_map_valid], axis=1)


# In[54]:


all_vars = cat_vars + contin_vars


# In[55]:


xgb_parms = {'learning_rate': 0.1, 'subsample': 0.6, 
             'colsample_bylevel': 0.6, 'silent': True, 'objective': 'reg:linear'}


# In[56]:


xdata = xgboost.DMatrix(X_train, y_train, feature_names=all_vars)


# In[57]:


xdata_val = xgboost.DMatrix(X_valid, y_valid, feature_names=all_vars)


# In[58]:


xgb_parms['seed'] = random.randint(0,1e9)
model = xgboost.train(xgb_parms, xdata)


# In[59]:


model.eval(xdata_val)


# In[60]:


model.eval(xdata_val)


# Easily, competition distance is the most important, while events are not important at all.
# 
# In real applications, putting together a feature importance plot is often a first step. Oftentimes, we can remove hundreds of thousands of features from consideration with importance plots. 

# In[61]:


importance = model.get_fscore()
importance = sorted(importance.items(), key=operator.itemgetter(1))

df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()

df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance');


# ## End
