import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from models import DnCNN, DnCNN_ResNet
from datasets import STL10
from utils import *


parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--resume", type=bool, default=False, help="Wether or not resume training")
parser.add_argument("--path_projection_matrix", type=str, default="../projection_matrices/thinL64_90", help='path to projection matrix')
parser.add_argument("--path_data", type=str, default="./STL10/", help='path to data')
parser.add_argument("--noise_min", type=float, default=1e-2, help="Noise level")
parser.add_argument("--noise_max", type=float, default=1e-1, help="Noise level")
parser.add_argument("--batch_size", type=int, default=10, help="Size of batch for training")
parser.add_argument("--num_of_layers", type=int, default=5, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=1, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
opt = parser.parse_args()

if not os.path.exists(opt.outf):
    os.mkdir(opt.outf)

A = load_projection_matrix(opt.path_projection_matrix)

transform_train = transforms.Compose([
            transforms.CenterCrop(64),
            transforms.Lambda(lambda x: noisy_fbp(x, A, opt.noise_min, opt.noise_max)),
            transforms.ToTensor(),])


def main():
    # Load data
    print("=> Load data...")
    root = './STL10/'
    data = STL10(opt.path_data, split='train', transform=transform_train, download=True)
    loader_train = torch.utils.data.DataLoader(data, batch_size=opt.batch_size, shuffle=True, num_workers=2)
    # Build model
    print("=> Build model...")
    net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    if opt.resume and os.path.exists(os.path.join(opt.outf, 'net.pth')):
        print("Resuming training.")
        net.load_state_dict(torch.load(os.path.join(opt.outf, 'net.pth')))
    else:
        print("Training from scratch.")
        net.apply(weights_init_kaiming)
    # Loss
    criterion = nn.MSELoss(size_average=False)
    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    # Training
    step = 0
    print("=> Begin training...")
    for epoch in range(opt.epochs):
        if epoch < opt.milestone:
            current_lr = opt.lr
        else:
            current_lr = opt.lr / 5.
        # Set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('Learning rate %f' % current_lr)
        # Train
        for i, (img_train, imgn_train) in enumerate(loader_train, 0):
            # training step
            net.train()
            net.zero_grad()
            optimizer.zero_grad()
            out_train = net(imgn_train.float())
            loss = criterion(out_train, img_train) / (imgn_train.size()[0]*2)
            loss.backward()
            optimizer.step()
            # Results
            net.eval()
            out_train = torch.clamp(out_train, 0., 1.)
            psnr_train = batch_PSNR(out_train, img_train, 1.)
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                (epoch+1, i+1, len(loader_train), loss.item(), psnr_train))
            step += 1

        # Save model
        torch.save(net.state_dict(), os.path.join(opt.outf, 'net.pth'))

        
if __name__ == "__main__":
    main()