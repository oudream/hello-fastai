
# coding: utf-8

# ## Neural style transfer

# Style transfer/ super resolution implementation in pytorch.

# In[1]:


get_ipython().magic(u'matplotlib inline')
import importlib
import utils2; importlib.reload(utils2)
from utils2 import *
from vgg16_avg import VGG16_Avg
from keras import metrics
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave


# In[2]:


import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.autograd import Variable
from torch.utils.serialization import load_lua
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets


# ### Setup

# In[3]:


path = '/data/jhoward/imagenet/sample/'
dpath = '/data/jhoward/fast/imagenet/sample/'


# In[4]:


fnames = pickle.load(open(dpath+'fnames.pkl', 'rb'))
n = len(fnames); n


# In[5]:


img=Image.open(fnames[50]); img


# In[6]:


rn_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((1,1,1,3))
preproc = lambda x: (x - rn_mean)[:, :, :, ::-1]


# In[7]:


img_arr = preproc(np.expand_dims(np.array(img), 0))
shp = img_arr.shape


# In[8]:


deproc = lambda x,s: np.clip(x.reshape(s)[:, :, :, ::-1] + rn_mean, 0, 255)


# ### Create model

# In[9]:


def download_convert_vgg16_model():
    model_url='http://cs.stanford.edu/people/jcjohns/fast-neural-style/models/vgg16.t7'
    file = get_file(model_url, cache_subdir='models')
    vgglua = load_lua(file).parameters()
    vgg = models.VGGFeature()
    for (src, dst) in zip(vgglua[0], vgg.parameters()): dst[:] = src[:]
    torch.save(vgg.state_dict(), dpath+'vgg16_feature.pth')


# In[10]:


url = 'https://s3-us-west-2.amazonaws.com/jcjohns-models/'
fname = 'vgg16-00b39a1b.pth'
file = get_file(fname, url+fname, cache_subdir='models')


# In[12]:


vgg = models.vgg.vgg16()
vgg.load_state_dict(torch.load(file))
optimizer = optim.Adam(vgg.parameters())


# In[14]:


vgg.cuda();


# In[15]:


arr_lr = bcolz.open(dpath+'trn_resized_72.bc')[:]
arr_hr = bcolz.open(dpath+'trn_resized_288.bc')[:]


# In[16]:


arr = bcolz.open(dpath+'trn_resized.bc')[:]


# In[17]:


x = Variable(arr[0])
y = model(x)


# In[18]:


url = 'http://www.platform.ai/models/'
fname = 'imagenet_class_index.json'
fpath = get_file(fname, url+fname, cache_subdir='models')


# In[19]:


class ResidualBlock(nn.Module):
    def __init__(self, num):
        super(ResidualBlock, self).__init__()
        self.c1 = nn.Conv2d(num, num, kernel_size=3, stride=1, padding=1)
        self.c2 = nn.Conv2d(num, num, kernel_size=3, stride=1, padding=1)
        self.b1 = nn.BatchNorm2d(num)
        self.b2 = nn.BatchNorm2d(num)

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return h + x


# In[22]:


class FastStyleNet(nn.Module):
    def __init__(self):
        super(FastStyleNet, self).__init__()
        self.cs = [nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=4)
            ,nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
            ,nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)]
        self.b1s = [nn.BatchNorm2d(i) for i in [32, 64, 128]]
        self.rs = [ResidualBlock(128) for i in range(5)]
        self.ds [nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
            ,nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)]
        self.b2s = [nn.BatchNorm2d(i) for i in [64, 32]]
        self.d3 = nn.Conv2d(32, 3, kernel_size=9, stride=1, padding=4)

    def forward(self, h):
        for i in range(3): h = F.relu(self.b1s[i](self.cs[i](x)))
        for r in self.rs: h = r(h)
        for i in range(2): h = F.relu(self.b2s[i](self.ds[i](x)))
        return self.d3(h)


# ### Loss functions and processing

# In[23]:


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    return features.bmm(features.transpose(1, 2)) / (ch * h * w)


# In[24]:


def vgg_preprocessing(batch):
    tensortype = type(batch.data)
    mean = tensortype(batch.data.size())
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    batch -= Variable(mean)


# In[25]:


def save_model(model, filename):
    state = model.state_dict()
    for key in state: state[key] = state[key].clone().cpu()
    torch.save(state, filename)


# In[26]:


def tensor_save_rgbimage(tensor, filename):
    img = tensor.clone().cpu().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img.save(filename)


# In[27]:


def tensor_save_bgrimage(tensor, filename):
    (b, g, r) = torch.chunk(tensor, 3)
    tensor = torch.cat((r, g, b))
    tensor_save_rgbimage(tensor, filename)


# In[28]:


def tensor_load_rgbimage(filename, size=None):
    img = Image.open(filename)
    if size is not None: img = img.resize((size, size), Image.ANTIALIAS)
    img = np.array(img).transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img


# In[29]:


def batch_rgb_to_bgr(batch):
    batch = batch.transpose(0, 1)
    (r, g, b) = torch.chunk(batch, 3)
    batch = torch.cat((b, g, r))
    batch = batch.transpose(0, 1)
    return batch


# In[30]:


def batch_bgr_to_rgb(batch):
    return batch_rgb_to_bgr(batch)


# ### Recreate input

# In[31]:


base = K.variable(img_arr)
gen_img = K.placeholder(shp)
batch = K.concatenate([base, gen_img],0)


# In[32]:


model = VGG16_Avg(input_tensor=batch, include_top=False)


# In[745]:


outputs = {l.name: l.output for l in model.layers}


# In[763]:


layer = outputs['block5_conv1']


# In[764]:


class Evaluator(object):
    def __init__(self, f, shp): 
        self.f = f
        self.shp = shp
        
    def loss(self, x):
        loss_, grads_ = self.f([x.reshape(self.shp)])
        self.grad_values = grads_.flatten().astype(np.float64)
        return loss_.astype(np.float64)

    def grads(self, x): return np.copy(self.grad_values)


# In[765]:


content_loss = lambda base, gen: metrics.mse(gen, base)
loss = content_loss(layer[0], layer[1])
grads = K.gradients(loss, gen_img)
fn = K.function([gen_img], [loss]+grads)


# In[766]:


evaluator = Evaluator(fn, shp)


# In[767]:


rand_img = lambda shape: np.random.uniform(-2.5, 2.5, shape)/100


# In[768]:


def solve_image(eval_obj, niter, x):
    for i in range(niter):
        x, min_val, info = fmin_l_bfgs_b(eval_obj.loss, x.flatten(),
                                         fprime=eval_obj.grads, maxfun=20)
        x = np.clip(x, -127,127)
        print('Current loss value:', min_val)
        imsave('{}res_at_iteration_{}.png'.format(path, i), deproc(x.copy(), shp)[0])
    return x


# In[769]:


iterations=10
x = rand_img(shp)


# In[770]:


x = solve_image(evaluator, iterations, x)


# conv 1 of last block (5)

# In[771]:


Image.open(path + 'res_at_iteration_9.png')


# conv 1 of 4th block

# In[762]:


Image.open(path + 'res_at_iteration_9.png')


# ### Recreate style

# In[700]:


def plot_arr(arr): plt.imshow(deproc(arr,arr.shape)[0].astype('uint8'))


# In[991]:


style = Image.open('data/starry_night.jpg')
style = style.resize(np.divide(style.size,3.5).astype('int32')); style


# In[953]:


style = Image.open('data/bird.jpg')
style = style.resize(np.divide(style.size,2.4).astype('int32')); style


# In[915]:


style = Image.open('data/simpsons.jpg')
style = style.resize(np.divide(style.size,2.7).astype('int32')); style


# In[992]:


w,h = style.size


# In[993]:


src = img_arr[:,:h,:w]
shp = src.shape
style_arr = preproc(np.expand_dims(style,0)[:,:,:,:3])
plot_arr(src)


# In[883]:


base = K.variable(style_arr)
gen_img = K.placeholder(shp)
batch = K.concatenate([base, gen_img],0)


# In[789]:


model = VGG16_Avg(input_tensor=batch, include_top=False)
outputs = {l.name: l.output for l in model.layers}


# In[790]:


layers = [outputs['block{}_conv1'.format(o)] for o in range(1,4)]


# In[791]:


def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    return K.dot(features, K.transpose(features)) / x.get_shape().num_elements()


# In[792]:


def style_loss(x, targ):
    return keras.metrics.mse(gram_matrix(x), gram_matrix(targ))


# In[793]:


loss = sum(style_loss(l[0], l[1]) for l in layers)
grads = K.gradients(loss, gen_img)
style_fn = K.function([gen_img], [loss]+grads)


# In[794]:


evaluator = Evaluator(style_fn, shp)


# In[799]:


iterations=10
x = rand_img(shp)


# In[800]:


x = solve_image(evaluator, iterations, x)


# In[784]:


Image.open(path + 'res_at_iteration_9.png')


# In[801]:


Image.open(path + 'res_at_iteration_9.png')


# ### Style transfer

# In[807]:


def total_variation_loss(x, r, c):
    assert K.ndim(x) == 3
    a = K.square(x[:r - 1, :c - 1, :] - x[1:, :c - 1, :])
    b = K.square(x[:r - 1, :c - 1, :] - x[:r - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


# In[994]:


base = K.variable(src)
style_v = K.variable(style_arr)
gen_img = K.placeholder(shp)
batch = K.concatenate([base, style_v, gen_img],0)


# In[995]:


model = VGG16_Avg(input_tensor=batch, include_top=False)
outputs = {l.name: l.output for l in model.layers}


# In[996]:


style_layers = [outputs['block{}_conv1'.format(o)] for o in range(1,6)]


# In[22]:


content_name = 'block4_conv2'


# In[1362]:


content_layer = outputs[content_name]


# In[998]:


input_layer = model.layers[0].output


# In[1004]:


loss = sum(style_loss(l[1], l[2]) for l in style_layers)
loss += content_loss(content_layer[0], content_layer[2])/10.
# loss += total_variation_loss(input_layer[2], h, w)/1e9
grads = K.gradients(loss, gen_img)
transfer_fn = K.function([gen_img], [loss]+grads)


# In[1005]:


evaluator = Evaluator(transfer_fn, shp)


# In[1006]:


iterations=10
x = rand_img(shp)/10.


# In[1010]:


x = solve_image(evaluator, iterations, x)


# In[1011]:


Image.open(path + 'res_at_iteration_9.png')


# In[932]:


Image.open(path + 'res_at_iteration_9.png')


# In[914]:


Image.open(path + 'res_at_iteration_9.png')


# ## Perceptual loss for Van Gogh

# In[62]:


inp_shape = (72,72,3)
inp = Input(inp_shape)


# In[ ]:


class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)
        
    def get_output_shape_for(self, s):
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')


# In[1224]:


ref_model = Model(inp, ReflectionPadding2D((60,20))(inp))
ref_model.compile('adam', 'mse')


# In[1225]:


p = ref_model.predict(arr_lr[50:51])


# In[1226]:


plt.imshow(p[0].astype('uint8'))


# In[65]:


def conv_block(x, filters, size, stride=(2,2), mode='same'):
    x = Convolution2D(filters, size, size, subsample=stride, border_mode=mode)(x)
    x = BatchNormalization(axis=1, mode=2)(x)
    return Activation('relu')(x)


# In[70]:


def res_block(ip, nf=64):
    x = conv_block(ip, nf, 3, (1,1))
    x = Convolution2D(nf, 3, 3, border_mode='same')(x)
    x = BatchNormalization(axis=1, mode=2)(x)
#     ip = Lambda(lambda x: x[:, 2:-2, 2:-2])(ip)
    return merge([x, ip], mode='sum')


# In[71]:


def deconv_block(x, filters, size, shape, stride=(2,2)):
    x = Deconvolution2D(filters, size, size, subsample=stride, border_mode='same',
                        output_shape=(None,)+shape)(x)
    x = BatchNormalization(axis=1, mode=2)(x)
    return Activation('relu')(x)


# In[72]:


parms = {'verbose': 0, 'callbacks': [TQDMNotebookCallback(leave_inner=True)]}


# In[97]:


inp=Input(inp_shape)
# x=ReflectionPadding2D((40, 40))(inp)
x=conv_block(inp, 64, 9, (1,1))
# x=conv_block(x, 64, 3)
# x=conv_block(x, 128, 3)
for i in range(4): x=res_block(x)
x=deconv_block(x, 64, 3, (144, 144, 64))
x=deconv_block(x, 64, 3, (288, 288, 64))
x=Convolution2D(3, 9, 9, activation='tanh', border_mode='same')(x)
outp=Lambda(lambda x: (x+1)*127.5)(x)


# In[98]:


vgg_l = Lambda(preproc)
outp_l = vgg_l(outp)


# In[99]:


out_shape = (288,288,3)
vgg_inp=Input(out_shape)
vgg= VGG16(include_top=False, input_tensor=vgg_l(vgg_inp))
for l in vgg.layers: l.trainable=False


# In[100]:


vgg_content = Model(vgg_inp, vgg.get_layer('block2_conv2').output)
vgg1 = vgg_content(vgg_inp)
vgg2 = vgg_content(outp)


# In[106]:


loss = Lambda(lambda x: K.sqrt(K.mean((x[0]-x[1])**2, (1,2))))([vgg1, vgg2])
m_final = Model([inp, vgg_inp], loss)
targ = np.zeros((arr_lr.shape[0], 128))


# In[107]:


m_final.compile('adam', 'mse')


# In[110]:


m_final.evaluate([arr_lr[:10],arr_hr[:10]], targ[:10])


# In[48]:


K.set_value(m_final.optimizer.lr, 1e-3)


# In[119]:


m_final.fit([arr_lr, arr_hr], targ, 8, 2, **parms)


# In[123]:


K.set_value(m_final.optimizer.lr, 1e-4)


# In[124]:


m_final.fit([arr_lr, arr_hr], targ, 16, 2, **parms)


# In[125]:


m_final.save_weights(dpath+'m_final.h5')


# In[126]:


top_model = Model(inp, outp)


# In[128]:


top_model.save_weights(dpath+'top_final.h5')


# In[135]:


p = top_model.predict(arr_lr[:20])


# In[133]:


plt.imshow(arr_lr[10].astype('uint8'));


# In[136]:


plt.imshow(p[10].astype('uint8'))


# In[117]:


plt.imshow(arr_hr[0].astype('uint8'));


# ### End
