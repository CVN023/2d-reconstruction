# Plug and Play ADMM

DnCNN PyTorch implementation of the [paper](https://arxiv.org/abs/1608.03981) *Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising*, inspired from this [repository](https://github.com/SaoYan/DnCNN-PyTorch).

Usage:

```
python train.py --path_projection_matrix 'path_projection_matrix' --path_data 'path_data' --noise_min 1e-2, --noise_max 1e-1 --batch_size 10 --num_of_layers 5 --epochs 3 --milestone 1 --lr 1e-3 --outf 'logs'
```
