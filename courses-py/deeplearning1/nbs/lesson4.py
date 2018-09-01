
# coding: utf-8

# In[1]:


from theano.sandbox import cuda


# In[2]:


get_ipython().magic(u'matplotlib inline')
import utils; reload(utils)
from utils import *
from __future__ import division, print_function


# In[33]:


#path = "data/ml-20m/"
path = "data/ml-small/"
model_path = path + 'models/'
if not os.path.exists(model_path): os.mkdir(model_path)
batch_size=64


# ## Set up data

# We're working with the movielens data, which contains one rating per row, like this:

# In[34]:


ratings = pd.read_csv(path+'ratings.csv')
ratings.head()


# In[35]:


len(ratings)


# Just for display purposes, let's read in the movie names too.

# In[36]:


movie_names = pd.read_csv(path+'movies.csv').set_index('movieId')['title'].to_dict()


# In[37]:


users = ratings.userId.unique()
movies = ratings.movieId.unique()


# In[38]:


userid2idx = {o:i for i,o in enumerate(users)}
movieid2idx = {o:i for i,o in enumerate(movies)}


# We update the movie and user ids so that they are contiguous integers, which we want when using embeddings.

# In[39]:


ratings.movieId = ratings.movieId.apply(lambda x: movieid2idx[x])
ratings.userId = ratings.userId.apply(lambda x: userid2idx[x])


# In[40]:


user_min, user_max, movie_min, movie_max = (ratings.userId.min(), 
    ratings.userId.max(), ratings.movieId.min(), ratings.movieId.max())
user_min, user_max, movie_min, movie_max


# In[41]:


n_users = ratings.userId.nunique()
n_movies = ratings.movieId.nunique()
n_users, n_movies


# This is the number of latent factors in each embedding.

# In[42]:


n_factors = 50


# In[43]:


np.random.seed = 42


# Randomly split into training and validation.

# In[44]:


msk = np.random.rand(len(ratings)) < 0.8
trn = ratings[msk]
val = ratings[~msk]


# ## Create subset for Excel

# We create a crosstab of the most popular movies and most movie-addicted users which we'll copy into Excel for creating a simple example. This isn't necessary for any of the modeling below however.

# In[76]:


g=ratings.groupby('userId')['rating'].count()
topUsers=g.sort_values(ascending=False)[:15]


# In[78]:


g=ratings.groupby('movieId')['rating'].count()
topMovies=g.sort_values(ascending=False)[:15]


# In[80]:


top_r = ratings.join(topUsers, rsuffix='_r', how='inner', on='userId')


# In[81]:


top_r = top_r.join(topMovies, rsuffix='_r', how='inner', on='movieId')


# In[82]:


pd.crosstab(top_r.userId, top_r.movieId, top_r.rating, aggfunc=np.sum)


# ## Dot product

# The most basic model is a dot product of a movie embedding and a user embedding. Let's see how well that works:

# In[23]:


user_in = Input(shape=(1,), dtype='int64', name='user_in')
u = Embedding(n_users, n_factors, input_length=1, W_regularizer=l2(1e-4))(user_in)
movie_in = Input(shape=(1,), dtype='int64', name='movie_in')
m = Embedding(n_movies, n_factors, input_length=1, W_regularizer=l2(1e-4))(movie_in)


# In[24]:


x = merge([u, m], mode='dot')
x = Flatten()(x)
model = Model([user_in, movie_in], x)
model.compile(Adam(0.001), loss='mse')


# In[25]:


model.fit([trn.userId, trn.movieId], trn.rating, batch_size=64, nb_epoch=1, 
          validation_data=([val.userId, val.movieId], val.rating))


# In[26]:


model.optimizer.lr=0.01


# In[27]:


model.fit([trn.userId, trn.movieId], trn.rating, batch_size=64, nb_epoch=3, 
          validation_data=([val.userId, val.movieId], val.rating))


# In[28]:


model.optimizer.lr=0.001


# In[29]:


model.fit([trn.userId, trn.movieId], trn.rating, batch_size=64, nb_epoch=6, 
          validation_data=([val.userId, val.movieId], val.rating))


# The [best benchmarks](http://www.librec.net/example.html) are a bit over 0.9, so this model doesn't seem to be working that well...

# ##  Bias

# The problem is likely to be that we don't have bias terms - that is, a single bias for each user and each movie representing how positive or negative each user is, and how good each movie is. We can add that easily by simply creating an embedding with one output for each movie and each user, and adding it to our output.

# In[45]:


def embedding_input(name, n_in, n_out, reg):
    inp = Input(shape=(1,), dtype='int64', name=name)
    return inp, Embedding(n_in, n_out, input_length=1, W_regularizer=l2(reg))(inp)


# In[46]:


user_in, u = embedding_input('user_in', n_users, n_factors, 1e-4)
movie_in, m = embedding_input('movie_in', n_movies, n_factors, 1e-4)


# In[47]:


def create_bias(inp, n_in):
    x = Embedding(n_in, 1, input_length=1)(inp)
    return Flatten()(x)


# In[48]:


ub = create_bias(user_in, n_users)
mb = create_bias(movie_in, n_movies)


# In[49]:


x = merge([u, m], mode='dot')
x = Flatten()(x)
x = merge([x, ub], mode='sum')
x = merge([x, mb], mode='sum')
model = Model([user_in, movie_in], x)
model.compile(Adam(0.001), loss='mse')


# In[418]:


model.fit([trn.userId, trn.movieId], trn.rating, batch_size=64, nb_epoch=1, 
          validation_data=([val.userId, val.movieId], val.rating))


# In[419]:


model.optimizer.lr=0.01


# In[420]:


model.fit([trn.userId, trn.movieId], trn.rating, batch_size=64, nb_epoch=6, 
          validation_data=([val.userId, val.movieId], val.rating))


# In[421]:


model.optimizer.lr=0.001


# In[422]:


model.fit([trn.userId, trn.movieId], trn.rating, batch_size=64, nb_epoch=10, 
          validation_data=([val.userId, val.movieId], val.rating))


# In[248]:


model.fit([trn.userId, trn.movieId], trn.rating, batch_size=64, nb_epoch=5, 
          validation_data=([val.userId, val.movieId], val.rating))


# This result is quite a bit better than the best benchmarks that we could find with a quick google search - so looks like a great approach!

# In[100]:


model.save_weights(model_path+'bias.h5')


# In[50]:


model.load_weights(model_path+'bias.h5')


# We can use the model to generate predictions by passing a pair of ints - a user id and a movie id. For instance, this predicts that user #3 would really enjoy movie #6.

# In[54]:


model.predict([np.array([3]), np.array([6])])


# ## Analyze results

# To make the analysis of the factors more interesting, we'll restrict it to the top 2000 most popular movies.

# In[56]:


g=ratings.groupby('movieId')['rating'].count()
topMovies=g.sort_values(ascending=False)[:2000]
topMovies = np.array(topMovies.index)


# First, we'll look at the movie bias term. We create a 'model' - which in keras is simply a way of associating one or more inputs with one more more outputs, using the functional API. Here, our input is the movie id (a single id), and the output is the movie bias (a single float).

# In[57]:


get_movie_bias = Model(movie_in, mb)
movie_bias = get_movie_bias.predict(topMovies)
movie_ratings = [(b[0], movie_names[movies[i]]) for i,b in zip(topMovies,movie_bias)]


# Now we can look at the top and bottom rated movies. These ratings are corrected for different levels of reviewer sentiment, as well as different types of movies that different reviewers watch.

# In[23]:


sorted(movie_ratings, key=itemgetter(0))[:15]


# In[24]:


sorted(movie_ratings, key=itemgetter(0), reverse=True)[:15]


# We can now do the same thing for the embeddings.

# In[59]:


get_movie_emb = Model(movie_in, m)
movie_emb = np.squeeze(get_movie_emb.predict([topMovies]))
movie_emb.shape


# Because it's hard to interpret 50 embeddings, we use [PCA](https://plot.ly/ipython-notebooks/principal-component-analysis/) to simplify them down to just 3 vectors. 

# In[27]:


from sklearn.decomposition import PCA
pca = PCA(n_components=3)
movie_pca = pca.fit(movie_emb.T).components_


# In[31]:


fac0 = movie_pca[0]


# In[32]:


movie_comp = [(f, movie_names[movies[i]]) for f,i in zip(fac0, topMovies)]


# Here's the 1st component. It seems to be 'critically acclaimed' or 'classic'.

# In[33]:


sorted(movie_comp, key=itemgetter(0), reverse=True)[:10]


# In[34]:


sorted(movie_comp, key=itemgetter(0))[:10]


# In[35]:


fac1 = movie_pca[1]


# In[36]:


movie_comp = [(f, movie_names[movies[i]]) for f,i in zip(fac1, topMovies)]


# The 2nd is 'hollywood blockbuster'.

# In[37]:


sorted(movie_comp, key=itemgetter(0), reverse=True)[:10]


# In[38]:


sorted(movie_comp, key=itemgetter(0))[:10]


# In[39]:


fac2 = movie_pca[2]


# In[40]:


movie_comp = [(f, movie_names[movies[i]]) for f,i in zip(fac2, topMovies)]


# The 3rd is 'violent vs happy'.

# In[41]:


sorted(movie_comp, key=itemgetter(0), reverse=True)[:10]


# In[42]:


sorted(movie_comp, key=itemgetter(0))[:10]


# We can draw a picture to see how various movies appear on the map of these components. This picture shows the 1st and 3rd components.

# In[401]:


import sys
stdout, stderr = sys.stdout, sys.stderr # save notebook stdout and stderr
reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdout, sys.stderr = stdout, stderr # restore notebook stdout and stderr


# In[410]:


start=50; end=100
X = fac0[start:end]
Y = fac2[start:end]
plt.figure(figsize=(15,15))
plt.scatter(X, Y)
for i, x, y in zip(topMovies[start:end], X, Y):
    plt.text(x,y,movie_names[movies[i]], color=np.random.rand(3)*0.7, fontsize=14)
plt.show()


# ##  Neural net

# Rather than creating a special purpose architecture (like our dot-product with bias earlier), it's often both easier and more accurate to use a standard neural network. Let's try it! Here, we simply concatenate the user and movie embeddings into a single vector, which we feed into the neural net.

# In[63]:


user_in, u = embedding_input('user_in', n_users, n_factors, 1e-4)
movie_in, m = embedding_input('movie_in', n_movies, n_factors, 1e-4)


# In[64]:


x = merge([u, m], mode='concat')
x = Flatten()(x)
x = Dropout(0.3)(x)
x = Dense(70, activation='relu')(x)
x = Dropout(0.75)(x)
x = Dense(1)(x)
nn = Model([user_in, movie_in], x)
nn.compile(Adam(0.001), loss='mse')


# In[65]:


nn.fit([trn.userId, trn.movieId], trn.rating, batch_size=64, nb_epoch=8, 
          validation_data=([val.userId, val.movieId], val.rating))


# This improves on our already impressive accuracy even further!
