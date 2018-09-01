
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ## Style transfer

# In[2]:


from fastai.conv_learner import *
from pathlib import Path
from scipy import ndimage
torch.cuda.set_device(3)

torch.backends.cudnn.benchmark=True


# In[3]:


PATH = Path('data/imagenet')
PATH_TRN = PATH/'train'


# In[4]:


m_vgg = to_gpu(vgg16(True)).eval()
set_trainable(m_vgg, False)


# In[5]:


img_fn = PATH_TRN/'n01558993'/'n01558993_9684.JPEG'
img = open_image(img_fn)
plt.imshow(img);


# In[6]:


sz=288


# In[7]:


trn_tfms,val_tfms = tfms_from_model(vgg16, sz)
img_tfm = val_tfms(img)
img_tfm.shape


# In[8]:


opt_img = np.random.uniform(0, 1, size=img.shape).astype(np.float32)
plt.imshow(opt_img);


# In[9]:


opt_img = scipy.ndimage.filters.median_filter(opt_img, [8,8,1])


# In[10]:


plt.imshow(opt_img);


# In[11]:


opt_img = val_tfms(opt_img)/2
opt_img_v = V(opt_img[None], requires_grad=True)
opt_img_v.shape


# In[12]:


m_vgg = nn.Sequential(*children(m_vgg)[:37])


# In[13]:


targ_t = m_vgg(VV(img_tfm[None]))
targ_v = V(targ_t)
targ_t.shape


# In[14]:


max_iter = 1000
show_iter = 100
optimizer = optim.LBFGS([opt_img_v], lr=0.5)


# In[15]:


def actn_loss(x): return F.mse_loss(m_vgg(x), targ_v)*1000


# In[37]:


def step(loss_fn):
    global n_iter
    optimizer.zero_grad()
    loss = loss_fn(opt_img_v)
    loss.backward()
    n_iter+=1
    if n_iter%show_iter==0: print(f'Iteration: n_iter, loss: {loss.data[0]}')
    return loss


# In[19]:


n_iter=0
while n_iter <= max_iter: optimizer.step(partial(step,actn_loss))


# In[22]:


x = val_tfms.denorm(np.rollaxis(to_np(opt_img_v.data),1,4))[0]
plt.figure(figsize=(7,7))
plt.imshow(x);


# ## forward hook

# In[39]:


class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = output
    def close(self): self.hook.remove()


# In[40]:


m_vgg = to_gpu(vgg16(True)).eval()
set_trainable(m_vgg, False)


# In[41]:


block_ends = [i-1 for i,o in enumerate(children(m_vgg))
              if isinstance(o,nn.MaxPool2d)]
block_ends


# In[42]:


sf = SaveFeatures(children(m_vgg)[block_ends[3]])


# In[43]:


def get_opt():
    opt_img = np.random.uniform(0, 1, size=img.shape).astype(np.float32)
    opt_img = scipy.ndimage.filters.median_filter(opt_img, [8,8,1])
    opt_img_v = V(val_tfms(opt_img/2)[None], requires_grad=True)
    return opt_img_v, optim.LBFGS([opt_img_v])


# In[44]:


opt_img_v, optimizer = get_opt()


# In[45]:


m_vgg(VV(img_tfm[None]))
targ_v = V(sf.features.clone())
targ_v.shape


# In[46]:


def actn_loss2(x):
    m_vgg(x)
    out = V(sf.features)
    return F.mse_loss(out, targ_v)*1000


# In[47]:


n_iter=0
while n_iter <= max_iter: optimizer.step(partial(step,actn_loss2))


# In[43]:


x = val_tfms.denorm(np.rollaxis(to_np(opt_img_v.data),1,4))[0]
plt.figure(figsize=(7,7))
plt.imshow(x);


# In[49]:


sf.close()


# ## Style match

# In[73]:


style_fn = PATH/'style'/'starry_night.jpg'


# In[74]:


style_img = open_image(style_fn)
style_img.shape, img.shape


# In[75]:


plt.imshow(style_img);


# In[76]:


def scale_match(src, targ):
    h,w,_ = src.shape
    sh,sw,_ = targ.shape
    rat = max(h/sh,w/sw); rat
    res = cv2.resize(targ, (int(sw*rat), int(sh*rat)))
    return res[:h,:w]


# In[77]:


style = scale_match(img, style_img)


# In[78]:


plt.imshow(style)
style.shape, img.shape


# In[79]:


opt_img_v, optimizer = get_opt()


# In[80]:


sfs = [SaveFeatures(children(m_vgg)[idx]) for idx in block_ends]


# In[81]:


m_vgg(VV(img_tfm[None]))
targ_vs = [V(o.features.clone()) for o in sfs]
[o.shape for o in targ_vs]


# In[82]:


style_tfm = val_tfms(style_img)


# In[83]:


m_vgg(VV(style_tfm[None]))
targ_styles = [V(o.features.clone()) for o in sfs]
[o.shape for o in targ_styles]


# In[84]:


def gram(input):
        b,c,h,w = input.size()
        x = input.view(b*c, -1)
        return torch.mm(x, x.t())/input.numel()*1e6

def gram_mse_loss(input, target): return F.mse_loss(gram(input), gram(target))


# In[85]:


def style_loss(x):
    m_vgg(opt_img_v)
    outs = [V(o.features) for o in sfs]
    losses = [gram_mse_loss(o, s) for o,s in zip(outs, targ_styles)]
    return sum(losses)


# In[65]:


n_iter=0
while n_iter <= max_iter: optimizer.step(partial(step,style_loss))


# In[59]:


x = val_tfms.denorm(np.rollaxis(to_np(opt_img_v.data),1,4))[0]
plt.figure(figsize=(7,7))
plt.imshow(x);


# In[68]:


for sf in sfs: sf.close()


# ## Style transfer

# In[86]:


opt_img_v, optimizer = get_opt()


# In[87]:


def comb_loss(x):
    m_vgg(opt_img_v)
    outs = [V(o.features) for o in sfs]
    losses = [gram_mse_loss(o, s) for o,s in zip(outs, targ_styles)]
    cnt_loss   = F.mse_loss(outs[3], targ_vs[3])*1000000
    style_loss = sum(losses)
    return cnt_loss + style_loss


# In[88]:


n_iter=0
while n_iter <= max_iter: optimizer.step(partial(step,comb_loss))


# In[69]:


x = val_tfms.denorm(np.rollaxis(to_np(opt_img_v.data),1,4))[0]
plt.figure(figsize=(9,9))
plt.imshow(x, interpolation='lanczos')
plt.axis('off');


# In[90]:


for sf in sfs: sf.close()

