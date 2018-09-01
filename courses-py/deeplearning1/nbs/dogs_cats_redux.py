
# coding: utf-8

# # Dogs vs Cat Redux

# In this tutorial, you will learn how generate and submit predictions to a Kaggle competiton
# 
# [Dogs vs. Cats Redux: Kernels Edition](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition)
#     
#     

# To start you will need to download and unzip the competition data from Kaggle and ensure your directory structure looks like this
# ```
# utils/
#     vgg16.py
#     utils.py
# lesson1/
#     redux.ipynb
#     data/
#         redux/
#             train/
#                 cat.437.jpg
#                 dog.9924.jpg
#                 cat.1029.jpg
#                 dog.4374.jpg
#             test/
#                 231.jpg
#                 325.jpg
#                 1235.jpg
#                 9923.jpg
# ```
# 
# You can download the data files from the competition page [here](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data) or you can download them from the command line using the [Kaggle CLI](https://github.com/floydwch/kaggle-cli).
# 
# You should launch your notebook inside the lesson1 directory
# ```
# cd lesson1
# jupyter notebook
# ```

# In[ ]:


#Verify we are in the lesson1 directory
get_ipython().magic(u'pwd')


# In[2]:


#Create references to important directories we will use over and over
import os, sys
current_dir = os.getcwd()
LESSON_HOME_DIR = current_dir
DATA_HOME_DIR = current_dir+'/data/redux'


# In[5]:


#Allow relative imports to directories above lesson1/
sys.path.insert(1, os.path.join(sys.path[0], '..'))

#import modules
from utils import *
from vgg16 import Vgg16

#Instantiate plotting tool
#In Jupyter notebooks, you will need to run this command before doing any plotting
get_ipython().magic(u'matplotlib inline')


# ## Action Plan
# 1. Create Validation and Sample sets
# 2. Rearrange image files into their respective directories 
# 3. Finetune and Train model
# 4. Generate predictions
# 5. Validate predictions
# 6. Submit predictions to Kaggle

# ## Create validation set and sample

# In[66]:


#Create directories
get_ipython().magic(u'cd $DATA_HOME_DIR')
get_ipython().magic(u'mkdir valid')
get_ipython().magic(u'mkdir results')
get_ipython().magic(u'mkdir -p sample/train')
get_ipython().magic(u'mkdir -p sample/test')
get_ipython().magic(u'mkdir -p sample/valid')
get_ipython().magic(u'mkdir -p sample/results')
get_ipython().magic(u'mkdir -p test/unknown')


# In[67]:


get_ipython().magic(u'cd $DATA_HOME_DIR/train')


# In[68]:


g = glob('*.jpg')
shuf = np.random.permutation(g)
for i in range(2000): os.rename(shuf[i], DATA_HOME_DIR+'/valid/' + shuf[i])


# In[69]:


from shutil import copyfile


# In[70]:


g = glob('*.jpg')
shuf = np.random.permutation(g)
for i in range(200): copyfile(shuf[i], DATA_HOME_DIR+'/sample/train/' + shuf[i])


# In[71]:


get_ipython().magic(u'cd $DATA_HOME_DIR/valid')


# In[72]:


g = glob('*.jpg')
shuf = np.random.permutation(g)
for i in range(50): copyfile(shuf[i], DATA_HOME_DIR+'/sample/valid/' + shuf[i])


# ## Rearrange image files into their respective directories

# In[73]:


#Divide cat/dog images into separate directories

get_ipython().magic(u'cd $DATA_HOME_DIR/sample/train')
get_ipython().magic(u'mkdir cats')
get_ipython().magic(u'mkdir dogs')
get_ipython().magic(u'mv cat.*.jpg cats/')
get_ipython().magic(u'mv dog.*.jpg dogs/')

get_ipython().magic(u'cd $DATA_HOME_DIR/sample/valid')
get_ipython().magic(u'mkdir cats')
get_ipython().magic(u'mkdir dogs')
get_ipython().magic(u'mv cat.*.jpg cats/')
get_ipython().magic(u'mv dog.*.jpg dogs/')

get_ipython().magic(u'cd $DATA_HOME_DIR/valid')
get_ipython().magic(u'mkdir cats')
get_ipython().magic(u'mkdir dogs')
get_ipython().magic(u'mv cat.*.jpg cats/')
get_ipython().magic(u'mv dog.*.jpg dogs/')

get_ipython().magic(u'cd $DATA_HOME_DIR/train')
get_ipython().magic(u'mkdir cats')
get_ipython().magic(u'mkdir dogs')
get_ipython().magic(u'mv cat.*.jpg cats/')
get_ipython().magic(u'mv dog.*.jpg dogs/')


# In[74]:


# Create single 'unknown' class for test set
get_ipython().magic(u'cd $DATA_HOME_DIR/test')
get_ipython().magic(u'mv *.jpg unknown/')


# ## Finetuning and Training

# In[75]:


get_ipython().magic(u'cd $DATA_HOME_DIR')

#Set path to sample/ path if desired
path = DATA_HOME_DIR + '/' #'/sample/'
test_path = DATA_HOME_DIR + '/test/' #We use all the test data
results_path=DATA_HOME_DIR + '/results/'
train_path=path + '/train/'
valid_path=path + '/valid/'


# In[76]:


#import Vgg16 helper class
vgg = Vgg16()


# In[77]:


#Set constants. You can experiment with no_of_epochs to improve the model
batch_size=64
no_of_epochs=3


# In[78]:


#Finetune the model
batches = vgg.get_batches(train_path, batch_size=batch_size)
val_batches = vgg.get_batches(valid_path, batch_size=batch_size*2)
vgg.finetune(batches)

#Not sure if we set this for all fits
vgg.model.optimizer.lr = 0.01


# In[79]:


#Notice we are passing in the validation dataset to the fit() method
#For each epoch we test our model against the validation set
latest_weights_filename = None
for epoch in range(no_of_epochs):
    print "Running epoch: %d" % epoch
    vgg.fit(batches, val_batches, nb_epoch=1)
    latest_weights_filename = 'ft%d.h5' % epoch
    vgg.model.save_weights(results_path+latest_weights_filename)
print "Completed %s fit operations" % no_of_epochs


# ## Generate Predictions

# Let's use our new model to make predictions on the test dataset

# In[80]:


batches, preds = vgg.test(test_path, batch_size = batch_size*2)


# In[81]:


#For every image, vgg.test() generates two probabilities 
#based on how we've ordered the cats/dogs directories.
#It looks like column one is cats and column two is dogs
print preds[:5]

filenames = batches.filenames
print filenames[:5]


# In[82]:


#You can verify the column ordering by viewing some images
from PIL import Image
Image.open(test_path + filenames[2])


# In[83]:


#Save our test results arrays so we can use them again later
save_array(results_path + 'test_preds.dat', preds)
save_array(results_path + 'filenames.dat', filenames)


# ## Validate Predictions

# Keras' *fit()* function conveniently shows us the value of the loss function, and the accuracy, after every epoch ("*epoch*" refers to one full run through all training examples). The most important metrics for us to look at are for the validation set, since we want to check for over-fitting. 
# 
# - **Tip**: with our first model we should try to overfit before we start worrying about how to reduce over-fitting - there's no point even thinking about regularization, data augmentation, etc if you're still under-fitting! (We'll be looking at these techniques shortly).
# 
# As well as looking at the overall metrics, it's also a good idea to look at examples of each of:
# 1. A few correct labels at random
# 2. A few incorrect labels at random
# 3. The most correct labels of each class (ie those with highest probability that are correct)
# 4. The most incorrect labels of each class (ie those with highest probability that are incorrect)
# 5. The most uncertain labels (ie those with probability closest to 0.5).

# Let's see what we can learn from these examples. (In general, this is a particularly useful technique for debugging problems in the model. However, since this model is so simple, there may not be too much to learn at this stage.)
# 
# Calculate predictions on validation set, so we can find correct and incorrect examples:

# In[84]:


vgg.model.load_weights(results_path+latest_weights_filename)


# In[85]:


val_batches, probs = vgg.test(valid_path, batch_size = batch_size)


# In[86]:


filenames = val_batches.filenames
expected_labels = val_batches.classes #0 or 1

#Round our predictions to 0/1 to generate labels
our_predictions = probs[:,0]
our_labels = np.round(1-our_predictions)


# In[1]:


from keras.preprocessing import image

#Helper function to plot images by index in the validation set 
#Plots is a helper function in utils.py
def plots_idx(idx, titles=None):
    plots([image.load_img(valid_path + filenames[i]) for i in idx], titles=titles)
    
#Number of images to view for each visualization task
n_view = 4


# In[88]:


#1. A few correct labels at random
correct = np.where(our_labels==expected_labels)[0]
print "Found %d correct labels" % len(correct)
idx = permutation(correct)[:n_view]
plots_idx(idx, our_predictions[idx])


# In[89]:


#2. A few incorrect labels at random
incorrect = np.where(our_labels!=expected_labels)[0]
print "Found %d incorrect labels" % len(incorrect)
idx = permutation(incorrect)[:n_view]
plots_idx(idx, our_predictions[idx])


# In[90]:


#3a. The images we most confident were cats, and are actually cats
correct_cats = np.where((our_labels==0) & (our_labels==expected_labels))[0]
print "Found %d confident correct cats labels" % len(correct_cats)
most_correct_cats = np.argsort(our_predictions[correct_cats])[::-1][:n_view]
plots_idx(correct_cats[most_correct_cats], our_predictions[correct_cats][most_correct_cats])


# In[91]:


#3b. The images we most confident were dogs, and are actually dogs
correct_dogs = np.where((our_labels==1) & (our_labels==expected_labels))[0]
print "Found %d confident correct dogs labels" % len(correct_dogs)
most_correct_dogs = np.argsort(our_predictions[correct_dogs])[:n_view]
plots_idx(correct_dogs[most_correct_dogs], our_predictions[correct_dogs][most_correct_dogs])


# In[92]:


#4a. The images we were most confident were cats, but are actually dogs
incorrect_cats = np.where((our_labels==0) & (our_labels!=expected_labels))[0]
print "Found %d incorrect cats" % len(incorrect_cats)
if len(incorrect_cats):
    most_incorrect_cats = np.argsort(our_predictions[incorrect_cats])[::-1][:n_view]
    plots_idx(incorrect_cats[most_incorrect_cats], our_predictions[incorrect_cats][most_incorrect_cats])


# In[93]:


#4b. The images we were most confident were dogs, but are actually cats
incorrect_dogs = np.where((our_labels==1) & (our_labels!=expected_labels))[0]
print "Found %d incorrect dogs" % len(incorrect_dogs)
if len(incorrect_dogs):
    most_incorrect_dogs = np.argsort(our_predictions[incorrect_dogs])[:n_view]
    plots_idx(incorrect_dogs[most_incorrect_dogs], our_predictions[incorrect_dogs][most_incorrect_dogs])


# In[94]:


#5. The most uncertain labels (ie those with probability closest to 0.5).
most_uncertain = np.argsort(np.abs(our_predictions-0.5))
plots_idx(most_uncertain[:n_view], our_predictions[most_uncertain])


# Perhaps the most common way to analyze the result of a classification model is to use a [confusion matrix](http://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/). Scikit-learn has a convenient function we can use for this purpose:

# In[95]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(expected_labels, our_labels)


# We can just print out the confusion matrix, or we can show a graphical view (which is mainly useful for dependents with a larger number of categories).

# In[96]:


plot_confusion_matrix(cm, val_batches.class_indices)


# ## Submit Predictions to Kaggle!

# Here's the format Kaggle requires for new submissions:
# ```
# imageId,isDog
# 1242, .3984
# 3947, .1000
# 4539, .9082
# 2345, .0000
# ```
# 
# Kaggle wants the imageId followed by the probability of the image being a dog. Kaggle uses a metric called [Log Loss](http://wiki.fast.ai/index.php/Log_Loss) to evaluate your submission.

# In[97]:


#Load our test predictions from file
preds = load_array(results_path + 'test_preds.dat')
filenames = load_array(results_path + 'filenames.dat')


# In[98]:


#Grab the dog prediction column
isdog = preds[:,1]
print "Raw Predictions: " + str(isdog[:5])
print "Mid Predictions: " + str(isdog[(isdog < .6) & (isdog > .4)])
print "Edge Predictions: " + str(isdog[(isdog == 1) | (isdog == 0)])


# [Log Loss](http://wiki.fast.ai/index.php/Log_Loss) doesn't support probability values of 0 or 1--they are undefined (and we have many). Fortunately, Kaggle helps us by offsetting our 0s and 1s by a very small value. So if we upload our submission now we will have lots of .99999999 and .000000001 values. This seems good, right?
# 
# Not so. There is an additional twist due to how log loss is calculated--log loss rewards predictions that are confident and correct (p=.9999,label=1), but it punishes predictions that are confident and wrong far more (p=.0001,label=1). See visualization below.

# In[128]:


#Visualize Log Loss when True value = 1
#y-axis is log loss, x-axis is probabilty that label = 1
#As you can see Log Loss increases rapidly as we approach 0
#But increases slowly as our predicted probability gets closer to 1
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import log_loss

x = [i*.0001 for i in range(1,10000)]
y = [log_loss([1],[[i*.0001,1-(i*.0001)]],eps=1e-15) for i in range(1,10000,1)]

plt.plot(x, y)
plt.axis([-.05, 1.1, -.8, 10])
plt.title("Log Loss when true label = 1")
plt.xlabel("predicted probability")
plt.ylabel("log loss")

plt.show()


# In[125]:


#So to play it safe, we use a sneaky trick to round down our edge predictions
#Swap all ones with .95 and all zeros with .05
isdog = isdog.clip(min=0.05, max=0.95)


# In[100]:


#Extract imageIds from the filenames in our test/unknown directory 
filenames = batches.filenames
ids = np.array([int(f[8:f.find('.')]) for f in filenames])


# Here we join the two columns into an array of [imageId, isDog]

# In[101]:


subm = np.stack([ids,isdog], axis=1)
subm[:5]


# In[102]:


get_ipython().magic(u'cd $DATA_HOME_DIR')
submission_file_name = 'submission1.csv'
np.savetxt(submission_file_name, subm, fmt='%d,%.5f', header='id,label', comments='')


# In[103]:


from IPython.display import FileLink
get_ipython().magic(u'cd $LESSON_HOME_DIR')
FileLink('data/redux/'+submission_file_name)


# You can download this file and submit on the Kaggle website or use the Kaggle command line tool's "submit" method.
