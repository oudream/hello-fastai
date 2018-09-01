
# coding: utf-8

# In[1]:


import ast

import pandas as pd

import datetime

from keras.layers import Input, Dense, Embedding, merge, Flatten, Merge, BatchNormalization
from keras.models import Model, load_model
from keras.regularizers import l2
import keras.backend as K
from keras.optimizers import SGD
import numpy as np

from sklearn.cluster import MeanShift, estimate_bandwidth

import utils

import data

from sklearn.model_selection import train_test_split

from bcolz_array_iterator import BcolzArrayIterator

import bcolz

from keras_tqdm import TQDMNotebookCallback
from keras.callbacks import ModelCheckpoint


# Below path is a shared directory, swap to own

# In[2]:


data_path = "/data/datasets/taxi/"


# ## Replication of 'csv_to_hdf5.py'

# Original repo used some bizarre tuple method of reading in data to save in a hdf5 file using fuel. The following does the same approach in that module, only using pandas and saving in a bcolz format (w/ training data as example)

# In[3]:


meta = pd.read_csv(data_path+'metaData_taxistandsID_name_GPSlocation.csv', header=0)


# In[66]:


meta.head()


# In[85]:


train = pd.read_csv(data_path+'train/train.csv', header=0)


# In[5]:


train.head()


# In[6]:


train['ORIGIN_CALL'] = pd.Series(pd.factorize(train['ORIGIN_CALL'])[0]) + 1


# In[7]:


train['ORIGIN_STAND']=pd.Series([0 if pd.isnull(x) or x=='' else int(x) for x in train["ORIGIN_STAND"]])


# In[8]:


train['TAXI_ID'] = pd.Series(pd.factorize(train['TAXI_ID'])[0]) + 1


# In[9]:


train['DAY_TYPE'] = pd.Series([ord(x[0]) - ord('A') for x in train['DAY_TYPE']])


# The array of long/lat coordinates per trip (row) is read in as a string. The function `ast.literal_eval(x)` evaluates the string into the expression it represents (safely). This happens below

# In[138]:


polyline = pd.Series([ast.literal_eval(x) for x in train['POLYLINE']])


# Split into latitude/longitude

# In[148]:


train['LATITUDE'] = pd.Series([np.array([point[1] for point in poly],dtype=np.float32) for poly in polyline])


# In[150]:


train['LONGITUDE'] = pd.Series([np.array([point[0] for point in poly],dtype=np.float32) for poly in polyline])


# In[157]:


utils.save_array(data_path+'train/train.bc', train.as_matrix())


# In[158]:


utils.save_array(data_path+'train/meta_train.bc', meta.as_matrix())


# ## Further Feature Engineering

# After converting 'csv_to_hdf5.py' functionality to pandas, I saved that array and then simply constructed the rest of the features as specified in the paper using pandas. I didn't bother seeing how the author did it as it was extremely obtuse and involved the fuel module.

# In[424]:


train = pd.DataFrame(utils.load_array(data_path+'train/train.bc'), columns=['TRIP_ID', 'CALL_TYPE', 'ORIGIN_CALL', 'ORIGIN_STAND', 'TAXI_ID',
       'TIMESTAMP', 'DAY_TYPE', 'MISSING_DATA', 'POLYLINE', 'LATITUDE', 'LONGITUDE'])


# In[425]:


train.head()


# The paper discusses how many categorical variables there are per category. The following all check out

# In[426]:


train['ORIGIN_CALL'].max()


# In[427]:


train['ORIGIN_STAND'].max()


# In[428]:


train['TAXI_ID'].max()


# Self-explanatory

# In[429]:


train['DAY_OF_WEEK'] = pd.Series([datetime.datetime.fromtimestamp(t).weekday() for t in train['TIMESTAMP']])


# Quarter hour of the day, i.e. 1 of the `4*24 = 96` quarter hours of the day

# In[430]:


train['QUARTER_HOUR'] = pd.Series([int((datetime.datetime.fromtimestamp(t).hour*60 + datetime.datetime.fromtimestamp(t).minute)/15)
                                   for t in train['TIMESTAMP']])


# Self-explanatory

# In[431]:


train['WEEK_OF_YEAR'] = pd.Series([datetime.datetime.fromtimestamp(t).isocalendar()[1] for t in train['TIMESTAMP']])


# Target coords are the last in the sequence (final position). If there are no positions, or only 1, then mark as invalid w/ nan in order to drop later

# In[433]:


train['TARGET'] = pd.Series([[l[1][0][-1], l[1][1][-1]] if len(l[1][0]) > 1 else numpy.nan for l in train[['LONGITUDE','LATITUDE']].iterrows()])


# This function creates the continuous inputs, which are the concatened k first and k last coords in a sequence, as discussed in the paper. 
# 
# If there aren't at least 2* k coords excluding the target, then the k first and k last overlap. In this case the sequence (excluding target) is padded at the end with the last coord in the sequence. The paper mentioned they padded front and back but didn't specify in what manner.
# 
# Also marks any invalid w/ na's

# In[437]:


def start_stop_inputs(k):
    result = []
    for l in train[['LONGITUDE','LATITUDE']].iterrows():
        if len(l[1][0]) < 2 or len(l[1][1]) < 2:
            result.append(numpy.nan)
        elif len(l[1][0][:-1]) >= 2*k:
            result.append(numpy.concatenate([l[1][0][0:k],l[1][0][-(k+1):-1],l[1][1][0:k],l[1][1][-(k+1):-1]]).flatten())
        else:
            l1 = numpy.lib.pad(l[1][0][:-1], (0,20-len(l[1][0][:-1])), mode='edge')
            l2 = numpy.lib.pad(l[1][1][:-1], (0,20-len(l[1][1][:-1])), mode='edge')
            result.append(numpy.concatenate([l1[0:k],l1[-k:],l2[0:k],l2[-k:]]).flatten())
    return pd.Series(result)        


# In[438]:


train['COORD_FEATURES'] = start_stop_inputs(5)


# In[442]:


train.shape


# In[441]:


train.dropna().shape


# Drop na's

# In[443]:


train = train.dropna()


# In[446]:


utils.save_array(data_path+'train/train_features.bc', train.as_matrix())


# ## End to end feature transformation

# In[155]:


train = pd.read_csv(data_path+'train/train.csv', header=0)


# In[ ]:


test = pd.read_csv(data_path+'test/test.csv', header=0)


# In[139]:


def start_stop_inputs(k, data, test):
    result = []
    for l in data[['LONGITUDE','LATITUDE']].iterrows():
        if not test:
            if len(l[1][0]) < 2 or len(l[1][1]) < 2:
                result.append(np.nan)
            elif len(l[1][0][:-1]) >= 2*k:
                result.append(np.concatenate([l[1][0][0:k],l[1][0][-(k+1):-1],l[1][1][0:k],l[1][1][-(k+1):-1]]).flatten())
            else:
                l1 = np.lib.pad(l[1][0][:-1], (0,4*k-len(l[1][0][:-1])), mode='edge')
                l2 = np.lib.pad(l[1][1][:-1], (0,4*k-len(l[1][1][:-1])), mode='edge')
                result.append(np.concatenate([l1[0:k],l1[-k:],l2[0:k],l2[-k:]]).flatten())
        else:
            if len(l[1][0]) < 1 or len(l[1][1]) < 1:
                result.append(np.nan)
            elif len(l[1][0]) >= 2*k:
                result.append(np.concatenate([l[1][0][0:k],l[1][0][-k:],l[1][1][0:k],l[1][1][-k:]]).flatten())
            else:
                l1 = np.lib.pad(l[1][0], (0,4*k-len(l[1][0])), mode='edge')
                l2 = np.lib.pad(l[1][1], (0,4*k-len(l[1][1])), mode='edge')
                result.append(np.concatenate([l1[0:k],l1[-k:],l2[0:k],l2[-k:]]).flatten())
    return pd.Series(result)     


# Pre-calculated below on train set

# In[143]:


lat_mean = 41.15731
lat_std = 0.074120656
long_mean = -8.6161413
long_std = 0.057200309


# In[ ]:


def feature_ext(data, test=False):   
    
    data['ORIGIN_CALL'] = pd.Series(pd.factorize(data['ORIGIN_CALL'])[0]) + 1

    data['ORIGIN_STAND']=pd.Series([0 if pd.isnull(x) or x=='' else int(x) for x in data["ORIGIN_STAND"]])

    data['TAXI_ID'] = pd.Series(pd.factorize(data['TAXI_ID'])[0]) + 1

    data['DAY_TYPE'] = pd.Series([ord(x[0]) - ord('A') for x in data['DAY_TYPE']])

    polyline = pd.Series([ast.literal_eval(x) for x in data['POLYLINE']])

    data['LATITUDE'] = pd.Series([np.array([point[1] for point in poly],dtype=np.float32) for poly in polyline])

    data['LONGITUDE'] = pd.Series([np.array([point[0] for point in poly],dtype=np.float32) for poly in polyline])
    
    if not test:
    
        data['TARGET'] = pd.Series([[l[1][0][-1], l[1][1][-1]] if len(l[1][0]) > 1 else np.nan for l in data[['LONGITUDE','LATITUDE']].iterrows()])

    
    data['LATITUDE'] = pd.Series([(t-lat_mean)/lat_std for t in data['LATITUDE']])
    
    data['LONGITUDE'] = pd.Series([(t-long_mean)/long_std for t in data['LONGITUDE']])
    
    data['COORD_FEATURES'] = start_stop_inputs(5, data, test)

    data['DAY_OF_WEEK'] = pd.Series([datetime.datetime.fromtimestamp(t).weekday() for t in data['TIMESTAMP']])

    data['QUARTER_HOUR'] = pd.Series([int((datetime.datetime.fromtimestamp(t).hour*60 + datetime.datetime.fromtimestamp(t).minute)/15)
                                       for t in data['TIMESTAMP']])

    data['WEEK_OF_YEAR'] = pd.Series([datetime.datetime.fromtimestamp(t).isocalendar()[1] for t in data['TIMESTAMP']])
    
        
    data = data.dropna()

    return data


# In[ ]:


train = feature_ext(train)


# In[ ]:


test = feature_ext(test, test=True)


# In[161]:


test.head()


# In[162]:


utils.save_array(data_path+'train/train_features.bc', train.as_matrix())


# In[163]:


utils.save_array(data_path+'test/test_features.bc', test.as_matrix())


# In[164]:


train.head()


# ## MEANSHIFT

# Meanshift clustering as performed in the paper

# In[ ]:


train = pd.DataFrame(utils.load_array(data_path+'train/train_features.bc'),columns=['TRIP_ID', 'CALL_TYPE', 'ORIGIN_CALL', 'ORIGIN_STAND', 'TAXI_ID',
       'TIMESTAMP', 'DAY_TYPE', 'MISSING_DATA', 'POLYLINE', 'LATITUDE', 'LONGITUDE', 'DAY_OF_WEEK',
                            'QUARTER_HOUR', "WEEK_OF_YEAR", "TARGET", "COORD_FEATURES"])


# Clustering performed on the targets

# In[532]:


y_targ = np.vstack(train["TARGET"].as_matrix())


# In[524]:


from sklearn.cluster import MeanShift, estimate_bandwidth


# Can use the commented out code for a estimate of bandwidth, which causes clustering to converge much quicker.
# 
# This is not mentioned in the paper but is included in the code. In order to get results similar to the paper's,
# they manually chose the uncommented bandwidth

# In[533]:


#bw = estimate_bandwidth(y_targ, quantile=.1, n_samples=1000)
bw = 0.001


# This takes some time

# In[545]:


ms = MeanShift(bandwidth=bw, bin_seeding=True, min_bin_freq=5)
ms.fit(y_targ)


# In[546]:


cluster_centers = ms.cluster_centers_


# This is very close to the number of clusters mentioned in the paper

# In[547]:


cluster_centers.shape


# In[548]:


utils.save_array(data_path+"cluster_centers_bw_001.bc", cluster_centers)


# ## Formatting Features for Bcolz iterator / garbage

# In[ ]:


train = pd.DataFrame(utils.load_array(data_path+'train/train_features.bc'),columns=['TRIP_ID', 'CALL_TYPE', 'ORIGIN_CALL', 'ORIGIN_STAND', 'TAXI_ID',
       'TIMESTAMP', 'DAY_TYPE', 'MISSING_DATA', 'POLYLINE', 'LATITUDE', 'LONGITUDE', 'TARGET',
                            'COORD_FEATURES', "DAY_OF_WEEK", "QUARTER_HOUR", "WEEK_OF_YEAR"])


# In[ ]:


cluster_centers = utils.load_array(data_path+"cluster_centers_bw_001.bc")


# In[50]:


long = np.array([c[0] for c in cluster_centers])
lat = np.array([c[1] for c in cluster_centers])


# In[ ]:


X_train, X_val = train_test_split(train, test_size=0.2, random_state=42)


# In[11]:


def get_features(data):
    return [np.vstack(data['COORD_FEATURES'].as_matrix()), np.vstack(data['ORIGIN_CALL'].as_matrix()), 
           np.vstack(data['TAXI_ID'].as_matrix()), np.vstack(data['ORIGIN_STAND'].as_matrix()),
           np.vstack(data['QUARTER_HOUR'].as_matrix()), np.vstack(data['DAY_OF_WEEK'].as_matrix()), 
           np.vstack(data['WEEK_OF_YEAR'].as_matrix()), np.array([long for i in range(0,data.shape[0])]),
               np.array([lat for i in range(0,data.shape[0])])]


# In[7]:


def get_target(data):
    return np.vstack(data["TARGET"].as_matrix())


# In[ ]:


X_train_features = get_features(X_train)


# In[14]:


X_train_target = get_target(X_train)


# In[13]:


utils.save_array(data_path+'train/X_train_features.bc', get_features(X_train))


# ## MODEL

# Load training data and cluster centers

# In[16]:


train = pd.DataFrame(utils.load_array(data_path+'train/train_features.bc'),columns=['TRIP_ID', 'CALL_TYPE', 'ORIGIN_CALL', 'ORIGIN_STAND', 'TAXI_ID',
       'TIMESTAMP', 'DAY_TYPE', 'MISSING_DATA', 'POLYLINE', 'LATITUDE', 'LONGITUDE', 'TARGET',
                            'COORD_FEATURES', "DAY_OF_WEEK", "QUARTER_HOUR", "WEEK_OF_YEAR"])


# Validation cuts 

# In[17]:


cuts = [
    1376503200, # 2013-08-14 18:00
    1380616200, # 2013-10-01 08:30
    1381167900, # 2013-10-07 17:45
    1383364800, # 2013-11-02 04:00
    1387722600  # 2013-12-22 14:30
]


# In[41]:


print(datetime.datetime.fromtimestamp(1376503200))


# In[22]:


train.shape


# In[24]:


val_indices = []
index = 0
for index, row in train.iterrows():
    time = row['TIMESTAMP']
    latitude = row['LATITUDE']
    for ts in cuts:
        if time <= ts and time + 15 * (len(latitude) - 1) >= ts:
            val_indices.append(index)
            break
    index += 1


# In[60]:


X_valid = train.iloc[val_indices]


# In[53]:


valid.head()


# In[35]:


for d in valid['TIMESTAMP']:
    print(datetime.datetime.fromtimestamp(d))


# In[58]:


X_train = train.drop(train.index[[val_indices]])


# In[5]:


cluster_centers = utils.load_array(data_path+"/data/cluster_centers_bw_001.bc")


# In[6]:


long = np.array([c[0] for c in cluster_centers])
lat = np.array([c[1] for c in cluster_centers])


# In[62]:


utils.save_array(data_path+'train/X_train.bc', X_train.as_matrix())


# In[64]:


utils.save_array(data_path+'valid/X_val.bc', X_valid.as_matrix())


# In[24]:


X_train = pd.DataFrame(utils.load_array(data_path+'train/X_train.bc'),columns=['TRIP_ID', 'CALL_TYPE', 'ORIGIN_CALL', 'ORIGIN_STAND', 'TAXI_ID',
       'TIMESTAMP', 'DAY_TYPE', 'MISSING_DATA', 'POLYLINE', 'LATITUDE', 'LONGITUDE', 'TARGET',
                            'COORD_FEATURES', "DAY_OF_WEEK", "QUARTER_HOUR", "WEEK_OF_YEAR"])


# In[25]:


X_val = pd.DataFrame(utils.load_array(data_path+'valid/X_val.bc'),columns=['TRIP_ID', 'CALL_TYPE', 'ORIGIN_CALL', 'ORIGIN_STAND', 'TAXI_ID',
       'TIMESTAMP', 'DAY_TYPE', 'MISSING_DATA', 'POLYLINE', 'LATITUDE', 'LONGITUDE', 'TARGET',
                            'COORD_FEATURES', "DAY_OF_WEEK", "QUARTER_HOUR", "WEEK_OF_YEAR"])


# The equirectangular loss function mentioned in the paper.
# 
# Note: Very important that y[0] is longitude and y[1] is latitude.
# 
# Omitted the radius of the earth constant "R" as it does not affect minimization and units were not given in the paper.

# In[7]:


def equirectangular_loss(y_true, y_pred):
    deg2rad = 3.141592653589793 / 180
    long_1 = y_true[:,0]*deg2rad
    long_2 = y_pred[:,0]*deg2rad
    lat_1 = y_true[:,1]*deg2rad
    lat_2 = y_pred[:,1]*deg2rad
    return 6371*K.sqrt(K.square((long_1 - long_2)*K.cos((lat_1 + lat_2)/2.))
                       +K.square(lat_1 - lat_2))


# In[9]:


def embedding_input(name, n_in, n_out, reg):
    inp = Input(shape=(1,), dtype='int64', name=name)
    return inp, Embedding(n_in, n_out, input_length=1, W_regularizer=l2(reg))(inp)


# The following returns a fully-connected model as mentioned in the paper. Takes as input k as defined before, and the cluster centers.
# 
# Inputs: Embeddings for each category, concatenated w/ the 4*k continous variable representing the first/last k coords as mentioned above.
# 
# Embeddings have no regularization, as it was not mentioned in paper, though are easily equipped to include.
# 
# Paper mentions global normalization. Didn't specify exactly how they did that, whether thay did it sequentially or whatnot. I just included a batchnorm layer for the continuous inputs.
# 
# After concatenation, 1 hidden layer of 500 neurons as called for in paper.
# 
# Finally, output layer has as many outputs as there are cluster centers, w/ a softmax activation. Call this output P.
# 
# The prediction is the weighted sum of each cluster center c_i w/ corresponding predicted prob P_i.
# 
# To facilitate this, dotted output w/ cluster latitudes and longitudes separately. (this happens at variable y), then concatenated 
#     into single tensor.
#     
# NOTE!!: You will see that I have the cluster center coords as inputs. Ideally, This function should store the cluster longs/lats as a constant to be used in the model, but I could not figure out. As a consequence, I pass them in as a repeated input.

# In[67]:


def taxi_mlp(k, cluster_centers):
    shp = cluster_centers.shape[0]
    nums = Input(shape=(4*k,))

    center_longs = Input(shape=(shp,))
    center_lats = Input(shape=(shp,))

    emb_names = ['client_ID', 'taxi_ID', "stand_ID", "quarter_hour", "day_of_week", "week_of_year"]
    emb_ins = [57106, 448, 64, 96, 7, 52]
    emb_outs = [10 for i in range(0,6)]
    regs = [0 for i in range(0,6)]

    embs = [embedding_input(e[0], e[1]+1, e[2], e[3]) for e in zip(emb_names, emb_ins, emb_outs, regs)]

    x = merge([nums] + [Flatten()(e[1]) for e in embs], mode='concat')

    x = Dense(500, activation='relu')(x)

    x = Dense(shp, activation='softmax')(x)

    y = merge([merge([x, center_longs], mode='dot'), merge([x, center_lats], mode='dot')], mode='concat')

    return Model(input = [nums]+[e[0] for e in embs] + [center_longs, center_lats], output = y)


# As mentioned, construction of repeated cluster longs/lats for input

# Iterator for in memory `train` pandas dataframe. I did this as opposed to bcolz iterator due to the pre-processing

# In[43]:


def data_iter(data, batch_size, cluster_centers):
    long = [c[0] for c in cluster_centers]
    lat = [c[1] for c in cluster_centers]
    i = 0
    N = data.shape[0]
    while True:
        yield ([np.vstack(data['COORD_FEATURES'][i:i+batch_size].as_matrix()), np.vstack(data['ORIGIN_CALL'][i:i+batch_size].as_matrix()), 
           np.vstack(data['TAXI_ID'][i:i+batch_size].as_matrix()), np.vstack(data['ORIGIN_STAND'][i:i+batch_size].as_matrix()),
           np.vstack(data['QUARTER_HOUR'][i:i+batch_size].as_matrix()), np.vstack(data['DAY_OF_WEEK'][i:i+batch_size].as_matrix()), 
           np.vstack(data['WEEK_OF_YEAR'][i:i+batch_size].as_matrix()), np.array([long for i in range(0,batch_size)]),
               np.array([lat for i in range(0,batch_size)])], np.vstack(data["TARGET"][i:i+batch_size].as_matrix()))
        i += batch_size


# In[ ]:


x=Lambda(thing)([x,long,lat])


# Of course, k in the model needs to match k from feature construction. We again use 5 as they did in the paper

# In[68]:


model = taxi_mlp(5, cluster_centers)


# Paper used SGD opt w/ following paramerters

# In[69]:


model.compile(optimizer=SGD(0.01, momentum=0.9), loss=equirectangular_loss, metrics=['mse'])


# In[73]:


X_train_feat = get_features(X_train)


# In[74]:


X_train_target = get_target(X_train)


# In[76]:


X_val_feat = get_features(X_valid)


# In[77]:


X_val_target = get_target(X_valid)


# In[78]:


tqdm = TQDMNotebookCallback()


# In[79]:


checkpoint = ModelCheckpoint(filepath=data_path+'models/tmp/weights.{epoch:03d}.{val_loss:.8f}.hdf5', save_best_only=True)


# In[80]:


batch_size=256


# ### original

# In[84]:


model.fit(X_train_feat, X_train_target, nb_epoch=1, batch_size=batch_size, validation_data=(X_val_feat, X_val_target), callbacks=[tqdm, checkpoint], verbose=0)


# In[ ]:


model.fit(X_train_feat, X_train_target, nb_epoch=30, batch_size=batch_size, validation_data=(X_val_feat, X_val_target), callbacks=[tqdm, checkpoint], verbose=0)


# In[20]:


model = load_model(data_path+'models/weights.0.0799.hdf5', custom_objects={'equirectangular_loss':equirectangular_loss})


# In[42]:


model.fit(X_train_feat, X_train_target, nb_epoch=100, batch_size=batch_size, validation_data=(X_val_feat, X_val_target), callbacks=[tqdm, checkpoint], verbose=0)


# In[43]:


model.save(data_path+'models/current_model.hdf5')


# ### new valid

# In[81]:


model.fit(X_train_feat, X_train_target, nb_epoch=1, batch_size=batch_size, validation_data=(X_val_feat, X_val_target), callbacks=[tqdm, checkpoint], verbose=0)


# In[ ]:


model.fit(X_train_feat, X_train_target, nb_epoch=400, batch_size=batch_size, validation_data=(X_val_feat, X_val_target), callbacks=[tqdm, checkpoint], verbose=0)


# In[102]:


model.save(data_path+'/models/current_model.hdf5')


# In[84]:


len(X_val_feat[0])


# It works, but it seems to converge unrealistically quick and the loss values are not the same. The paper does not mention what it's using as "error" in it's results. I assume the same equirectangular? Not very clear. The difference in values could be due to the missing Earth-radius factor

# ## Kaggle Entry

# In[23]:


best_model = load_model(data_path+'models/weights.308.0.03373993.hdf5', custom_objects={'equirectangular_loss':equirectangular_loss})


# In[104]:


best_model.evaluate(X_val_feat, X_val_target)


# In[61]:


test = pd.DataFrame(utils.load_array(data_path+'test/test_features.bc'),columns=['TRIP_ID', 'CALL_TYPE', 'ORIGIN_CALL', 'ORIGIN_STAND', 'TAXI_ID',
       'TIMESTAMP', 'DAY_TYPE', 'MISSING_DATA', 'POLYLINE', 'LATITUDE', 'LONGITUDE',
                            'COORD_FEATURES', "DAY_OF_WEEK", "QUARTER_HOUR", "WEEK_OF_YEAR"])


# In[62]:


test['ORIGIN_CALL'] = pd.read_csv(data_path+'real_origin_call.csv', header=None)


# In[63]:


test['TAXI_ID'] = pd.read_csv(data_path+'real_taxi_id.csv',header=None)


# In[64]:


X_test = get_features(test)


# In[65]:


b = np.sort(X_test[1],axis=None)


# In[67]:


test_preds = np.round(best_model.predict(X_test), decimals=6)


# In[68]:


d = {0:test['TRIP_ID'], 1:test_preds[:,1], 2:test_preds[:,0]}
kaggle_out = pd.DataFrame(data=d)


# In[121]:


kaggle_out.to_csv(data_path+'submission.csv', header=['TRIP_ID','LATITUDE', 'LONGITUDE'], index=False)


# In[117]:


def hdist(a, b):
    deg2rad = 3.141592653589793 / 180

    lat1 = a[:, 1] * deg2rad
    lon1 = a[:, 0] * deg2rad
    lat2 = b[:, 1] * deg2rad
    lon2 = b[:, 0] * deg2rad

    dlat = abs(lat1-lat2)
    dlon = abs(lon1-lon2)

    al = np.sin(dlat/2)**2  + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon/2)**2)
    d = np.arctan2(np.sqrt(al), np.sqrt(1-al))

    hd = 2 * 6371 * d

    return hd


# In[118]:


val_preds = best_model.predict(X_val_feat)


# In[88]:


trn_preds = model.predict(X_train_feat)


# In[119]:


er = hdist(val_preds, X_val_target)


# In[120]:


er.mean()


# In[ ]:


K.equal()


# To-do: simple to extend to validation data

# ## Uh oh... training data not representative of test

# In[67]:


cuts = [
    1376503200, # 2013-08-14 18:00
    1380616200, # 2013-10-01 08:30
    1381167900, # 2013-10-07 17:45
    1383364800, # 2013-11-02 04:00
    1387722600  # 2013-12-22 14:30
]


# In[86]:


np.any([train['TIMESTAMP'].map(lambda x: x in cuts)])


# In[87]:


train['TIMESTAMP']


# In[90]:


np.any(train['TIMESTAMP']==1381167900)


# In[91]:


times = train['TIMESTAMP'].as_matrix()


# In[98]:


X_train.columns


# In[92]:


times


# In[102]:



count = 0
for index, row in X_val.iterrows():
    for ts in cuts:
        time = row['TIMESTAMP']
        latitude = row['LATITUDE']
        if time <= ts and time + 15 * (len(latitude) - 1) >= ts:
            count += 1


# In[101]:


one = count


# In[104]:


count + one


# In[6]:


import h5py


# In[7]:


h = h5py.File(data_path+'original/data.hdf5', 'r')


# In[15]:


evrData=h['/Configure:0000/Run:0000/CalibCycle:0000/EvrData::DataV3/NoDetector.0:Evr.0/data']


# In[13]:


c = np.load(data_path+'original/arrival-clusters.pkl')


# ### hd5f files

# In[10]:


from fuel.utils import find_in_data_path
from fuel.datasets import H5PYDataset


# In[7]:


original_path = '/data/bckenstler/data/taxi/original/'


# In[33]:


train_set = H5PYDataset(original_path+'data.hdf5', which_sets=('train',),load_in_memory=True)


# In[48]:


valid_set = H5PYDataset(original_path+'valid.hdf5', which_sets=('cuts/test_times_0',),load_in_memory=True)


# In[34]:


print(train_set.num_examples)


# In[28]:


print(valid_set.num_examples)


# In[37]:


data = train_set.data_sources


# In[44]:


data[0]


# In[49]:


valid_data = valid_set.data_sources


# In[89]:


valid_data[4][0]


# In[77]:


stamps = valid_data[-3]


# In[99]:


stamps[0]


# In[115]:


for i in range(0,304):    
    print(np.any([t==int(stamps[i]) for t in X_val['TIMESTAMP']]))


# In[101]:


type(X_train['TIMESTAMP'][0])


# In[83]:


type(stamps[0])


# In[78]:


check = [s in stamps for s in X_val['TIMESTAMP']]


# In[86]:


for s in X_val['TIMESTAMP']:
    print(datetime.datetime.fromtimestamp(s))


# In[85]:


for s in stamps:
    print(datetime.datetime.fromtimestamp(s))


# In[71]:


ids = valid_data[-1]


# In[74]:


type(ids[0])


# In[70]:


ids


# In[64]:


X_val

