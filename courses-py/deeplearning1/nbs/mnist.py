
# coding: utf-8

# In[1]:


from theano.sandbox import cuda
cuda.use('gpu2')


# In[2]:


get_ipython().magic(u'matplotlib inline')
import utils; reload(utils)
from utils import *
from __future__ import division, print_function


# ## Setup

# In[3]:


batch_size=64


# In[4]:


from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# In[5]:


X_test = np.expand_dims(X_test,1)
X_train = np.expand_dims(X_train,1)


# In[6]:


X_train.shape


# In[7]:


y_train[:5]


# In[8]:


y_train = onehot(y_train)
y_test = onehot(y_test)


# In[9]:


y_train[:5]


# In[10]:


mean_px = X_train.mean().astype(np.float32)
std_px = X_train.std().astype(np.float32)


# In[11]:


def norm_input(x): return (x-mean_px)/std_px


# ## Linear model

# In[160]:


def get_lin_model():
    model = Sequential([
        Lambda(norm_input, input_shape=(1,28,28)),
        Flatten(),
        Dense(10, activation='softmax')
        ])
    model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# In[161]:


lm = get_lin_model()


# In[162]:


gen = image.ImageDataGenerator()
batches = gen.flow(X_train, y_train, batch_size=64)
test_batches = gen.flow(X_test, y_test, batch_size=64)


# In[164]:


lm.fit_generator(batches, batches.N, nb_epoch=1, 
                    validation_data=test_batches, nb_val_samples=test_batches.N)


# In[167]:


lm.optimizer.lr=0.1


# In[169]:


lm.fit_generator(batches, batches.N, nb_epoch=1, 
                    validation_data=test_batches, nb_val_samples=test_batches.N)


# In[172]:


lm.optimizer.lr=0.01


# In[173]:


lm.fit_generator(batches, batches.N, nb_epoch=4, 
                    validation_data=test_batches, nb_val_samples=test_batches.N)


# ## Single dense layer

# In[175]:


def get_fc_model():
    model = Sequential([
        Lambda(norm_input, input_shape=(1,28,28)),
        Flatten(),
        Dense(512, activation='softmax'),
        Dense(10, activation='softmax')
        ])
    model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# In[182]:


fc = get_fc_model()


# In[183]:


fc.fit_generator(batches, batches.N, nb_epoch=1, 
                    validation_data=test_batches, nb_val_samples=test_batches.N)


# In[184]:


fc.optimizer.lr=0.1


# In[185]:


fc.fit_generator(batches, batches.N, nb_epoch=4, 
                    validation_data=test_batches, nb_val_samples=test_batches.N)


# In[187]:


fc.optimizer.lr=0.01


# In[189]:


fc.fit_generator(batches, batches.N, nb_epoch=4, 
                    validation_data=test_batches, nb_val_samples=test_batches.N)


# ## Basic 'VGG-style' CNN

# In[14]:


def get_model():
    model = Sequential([
        Lambda(norm_input, input_shape=(1,28,28)),
        Convolution2D(32,3,3, activation='relu'),
        Convolution2D(32,3,3, activation='relu'),
        MaxPooling2D(),
        Convolution2D(64,3,3, activation='relu'),
        Convolution2D(64,3,3, activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(10, activation='softmax')
        ])
    model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# In[45]:


model = get_model()


# In[36]:


model.fit_generator(batches, batches.N, nb_epoch=1, 
                    validation_data=test_batches, nb_val_samples=test_batches.N)


# In[37]:


model.optimizer.lr=0.1


# In[38]:


model.fit_generator(batches, batches.N, nb_epoch=1, 
                    validation_data=test_batches, nb_val_samples=test_batches.N)


# In[39]:


model.optimizer.lr=0.01


# In[40]:


model.fit_generator(batches, batches.N, nb_epoch=8, 
                    validation_data=test_batches, nb_val_samples=test_batches.N)


# ## Data augmentation

# In[23]:


model = get_model()


# In[76]:


gen = image.ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                               height_shift_range=0.08, zoom_range=0.08)
batches = gen.flow(X_train, y_train, batch_size=64)
test_batches = gen.flow(X_test, y_test, batch_size=64)


# In[24]:


model.fit_generator(batches, batches.N, nb_epoch=1, 
                    validation_data=test_batches, nb_val_samples=test_batches.N)


# In[25]:


model.optimizer.lr=0.1


# In[26]:


model.fit_generator(batches, batches.N, nb_epoch=4, 
                    validation_data=test_batches, nb_val_samples=test_batches.N)


# In[27]:


model.optimizer.lr=0.01


# In[28]:


model.fit_generator(batches, batches.N, nb_epoch=8, 
                    validation_data=test_batches, nb_val_samples=test_batches.N)


# In[29]:


model.optimizer.lr=0.001


# In[30]:


model.fit_generator(batches, batches.N, nb_epoch=14, 
                    validation_data=test_batches, nb_val_samples=test_batches.N)


# In[31]:


model.optimizer.lr=0.0001


# In[32]:


model.fit_generator(batches, batches.N, nb_epoch=10, 
                    validation_data=test_batches, nb_val_samples=test_batches.N)


# ## Batchnorm + data augmentation

# In[125]:


def get_model_bn():
    model = Sequential([
        Lambda(norm_input, input_shape=(1,28,28)),
        Convolution2D(32,3,3, activation='relu'),
        BatchNormalization(axis=1),
        Convolution2D(32,3,3, activation='relu'),
        MaxPooling2D(),
        BatchNormalization(axis=1),
        Convolution2D(64,3,3, activation='relu'),
        BatchNormalization(axis=1),
        Convolution2D(64,3,3, activation='relu'),
        MaxPooling2D(),
        Flatten(),
        BatchNormalization(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dense(10, activation='softmax')
        ])
    model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# In[126]:


model = get_model_bn()


# In[127]:


model.fit_generator(batches, batches.N, nb_epoch=1, 
                    validation_data=test_batches, nb_val_samples=test_batches.N)


# In[128]:


model.optimizer.lr=0.1


# In[129]:


model.fit_generator(batches, batches.N, nb_epoch=4, 
                    validation_data=test_batches, nb_val_samples=test_batches.N)


# In[130]:


model.optimizer.lr=0.01


# In[131]:


model.fit_generator(batches, batches.N, nb_epoch=12, 
                    validation_data=test_batches, nb_val_samples=test_batches.N)


# In[132]:


model.optimizer.lr=0.001


# In[133]:


model.fit_generator(batches, batches.N, nb_epoch=12, 
                    validation_data=test_batches, nb_val_samples=test_batches.N)


# ## Batchnorm + dropout + data augmentation

# In[79]:


def get_model_bn_do():
    model = Sequential([
        Lambda(norm_input, input_shape=(1,28,28)),
        Convolution2D(32,3,3, activation='relu'),
        BatchNormalization(axis=1),
        Convolution2D(32,3,3, activation='relu'),
        MaxPooling2D(),
        BatchNormalization(axis=1),
        Convolution2D(64,3,3, activation='relu'),
        BatchNormalization(axis=1),
        Convolution2D(64,3,3, activation='relu'),
        MaxPooling2D(),
        Flatten(),
        BatchNormalization(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(10, activation='softmax')
        ])
    model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# In[80]:


model = get_model_bn_do()


# In[81]:


model.fit_generator(batches, batches.N, nb_epoch=1, 
                    validation_data=test_batches, nb_val_samples=test_batches.N)


# In[82]:


model.optimizer.lr=0.1


# In[83]:


model.fit_generator(batches, batches.N, nb_epoch=4, 
                    validation_data=test_batches, nb_val_samples=test_batches.N)


# In[84]:


model.optimizer.lr=0.01


# In[85]:


model.fit_generator(batches, batches.N, nb_epoch=12, 
                    validation_data=test_batches, nb_val_samples=test_batches.N)


# In[86]:


model.optimizer.lr=0.001


# In[89]:


model.fit_generator(batches, batches.N, nb_epoch=1, 
                    validation_data=test_batches, nb_val_samples=test_batches.N)


# ## Ensembling

# In[90]:


def fit_model():
    model = get_model_bn_do()
    model.fit_generator(batches, batches.N, nb_epoch=1, verbose=0,
                        validation_data=test_batches, nb_val_samples=test_batches.N)
    model.optimizer.lr=0.1
    model.fit_generator(batches, batches.N, nb_epoch=4, verbose=0,
                        validation_data=test_batches, nb_val_samples=test_batches.N)
    model.optimizer.lr=0.01
    model.fit_generator(batches, batches.N, nb_epoch=12, verbose=0,
                        validation_data=test_batches, nb_val_samples=test_batches.N)
    model.optimizer.lr=0.001
    model.fit_generator(batches, batches.N, nb_epoch=18, verbose=0,
                        validation_data=test_batches, nb_val_samples=test_batches.N)
    return model


# In[91]:


models = [fit_model() for i in range(6)]


# In[92]:


path = "data/mnist/"
model_path = path + 'models/'


# In[93]:


for i,m in enumerate(models):
    m.save_weights(model_path+'cnn-mnist23-'+str(i)+'.pkl')


# In[94]:


evals = np.array([m.evaluate(X_test, y_test, batch_size=256) for m in models])


# In[95]:


evals.mean(axis=0)


# In[96]:


all_preds = np.stack([m.predict(X_test, batch_size=256) for m in models])


# In[97]:


all_preds.shape


# In[98]:


avg_preds = all_preds.mean(axis=0)


# In[99]:


keras.metrics.categorical_accuracy(y_test, avg_preds).eval()

