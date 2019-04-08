''' PyTorch implementaion of "Beyond a Gaussian Denoiser:
Residual Learning of Deep CNN for Image Denoising" (TIP2017)
Inspired from https://github.com/SaoYan/DnCNN-PyTorch '''

import torch
import torch.nn as nn



class DnCNN(nn.Module):

    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        eps = self.dncnn(x)
        return eps # out




class DnCNN_ResNet(nn.Module):

    def __init__(self, channels, num_of_layers=17):
        super(DnCNN_ResNet, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        self.in_block = nn.Sequential(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False),
                                      nn.ReLU(inplace=True))
        self.med_blocks = []
        for i in range(num_of_layers-2):                 
            self.med_blocks.append(nn.Sequential(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False),
                                                 nn.BatchNorm2d(features),
                                                 nn.ReLU(inplace=True)))
        self.out_block = nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        x = x + self.in_block(x)
        for i in range(len(self.med_blocks)):
            x = x + self.med_blocks[i](x)
        return self.out_block(x)