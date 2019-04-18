''' Utilities for PyTorch implementaion of "Beyond a Gaussian Denoiser:
Residual Learning of Deep CNN for Image Denoising" (TIP2017)
Inspired from https://github.com/SaoYan/DnCNN-PyTorch '''

import math
import torch
import torch.nn as nn
import numpy as np
from skimage.measure.simple_metrics import compare_psnr
import scipy.io
from skimage.transform import iradon



def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant_(m.bias.data, 0.0)


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])



def load_projection_matrix(path):
    mat = scipy.io.loadmat(path)
    return mat['H'].todense()


def noisy_fbp(x, A, noise_min=1e2, noise_max=1e3):
    # Gray scale
    x = x.convert('LA')
    x = 1 - np.array(x)[:,:,0]
    # Projection
    p = np.matmul(A, x.flatten())
    # Add noise
    noises = np.random.uniform(noise_min, noise_max, size=p.shape)
    p = p + np.random.normal(np.zeros(p.shape), noises)
    return np.rot90(iradon(p.reshape(90, 90).T, output_size=64, circle=False)).copy()

