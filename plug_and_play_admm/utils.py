import numpy as np
import torch


def denoising_image(image, net):
    input_ = np.expand_dims(np.expand_dims(image, 0), 0)
    input_ = torch.from_numpy(input_.copy())
    return net(input_.float()).detach().numpy()[0,0,:,:]



def grad(f):
    S = f.shape
    s0 = np.concatenate( (np.arange(1,S[0]),[0]) )
    s1 = np.concatenate( (np.arange(1,S[1]),[0]) )
    g = np.dstack( (f[s0,:] - f, f[:,s1] - f))
    return g


def div(g):
    S = g.shape
    s0 = np.concatenate( ([S[0]-1], np.arange(0,S[0]-1)) )
    s1 = np.concatenate( ([S[1]-1], np.arange(0,S[1]-1)) )
    f = (g[:,:,0] - g[s0,:,0]) + (g[:,:,1]-g[:,s1,1])
    return f
