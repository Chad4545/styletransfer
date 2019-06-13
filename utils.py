#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from PIL import Image


def load_image(filename,size=None, scale=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size,size),Image.BICUBIC)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale),int(img.size[1]/scale)),Image.BICUBIC)
    return img

def save_image(filename, data):
    # data == torch
    # clone == 복사
    # Clamp all elements in input into the range [min, max] and return a resulting Tensor.
    img = data.clone().clamp(0,255).numpy()
    # tensor: (C, W, H) - > np: (W,H,C)
    img = img.transpose(1,2,0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)
    
def gram_matrix(y):
    (b,ch,h,w) = y.size()
    
    features = y.view(b,ch,w*h)
    #b,c, 65536
    features_t = features.transpose(1,2)
    #b, 65536, c
    gram = features.bmm(features_t)/(ch*h*w)
    # b,c,c
    return gram

def normalize_batch(batch):
    # normalize using imagenet mean and std

    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    # 3, 1, 1
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    # 3,1,1
    batch = batch.div_(255.0)
    return (batch - mean) / std

