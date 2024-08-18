import os
from tqdm import tqdm
import uuid
import torch
import airbench
from utils import get_loaders

# Training script adapted from https://github.com/MadryLab/datamodels/blob/main/examples/cifar10/train_cifar.py
import numpy as np
import torch as ch
import torch.nn.functional as F
# Model (from KakaoBrain: https://github.com/wbaek/torchskeleton)
class Mul(ch.nn.Module):
    def __init__(self, weight):
        super(Mul, self).__init__()
        self.weight = weight
    def forward(self, x): return x * self.weight

class Flatten(ch.nn.Module):
    def forward(self, x): return x.view(x.size(0), -1) 

class Residual(ch.nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module
    def forward(self, x): return x + self.module(x)

def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1):
    return ch.nn.Sequential(
            ch.nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size,
                         stride=stride, padding=padding, groups=groups, bias=False),
            ch.nn.BatchNorm2d(channels_out),
            ch.nn.ReLU(inplace=True)
    )   

def construct_model():
    num_class = 10
    model = ch.nn.Sequential(
        conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
        conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
        Residual(ch.nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))),
        conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
        ch.nn.MaxPool2d(2),
        Residual(ch.nn.Sequential(conv_bn(256, 256), conv_bn(256, 256))),
        conv_bn(256, 128, kernel_size=3, stride=1, padding=0),
        ch.nn.AdaptiveMaxPool2d((1, 1)),
        Flatten(),
        ch.nn.Linear(128, num_class, bias=False),
        Mul(0.2)
    )   
    model = model.to(memory_format=ch.channels_last).cuda()
    return model

def train(loader):

    lr = 0.5 
    epochs = 24
    lr_peak_epoch = 5
    momentum = 0.9
    weight_decay = 5e-4
    # label_smoothing = 0.1
    label_smoothing = 0.0

    model = construct_model()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    iters_per_epoch = len(loader)
    # Cyclic LR with single triangle
    lr_schedule = np.interp(np.arange((epochs+1) * iters_per_epoch),
                            [0, lr_peak_epoch * iters_per_epoch, epochs * iters_per_epoch],
                            [0, 1, 0])
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)
    scaler = torch.cuda.amp.GradScaler()

    for _ in range(epochs):
        for ims, labs in loader:
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                out = model(ims)
                loss = F.cross_entropy(out, labs)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()

    return model.half()


if __name__ == '__main__':

    loader_all, loader_minus40 = get_loaders()

    os.makedirs('nets3_default', exist_ok=True)
    for _ in tqdm(range(500)):
        net = train(loader_all)
        outs = airbench.infer(net, loader_all)
        torch.save(outs, 'nets3_default/%s.pt' % uuid.uuid4())

    os.makedirs('nets1_minus_n40', exist_ok=True)
    for _ in tqdm(range(500)):
        net = train(loader_minus40)
        outs = airbench.infer(net, loader_all)
        torch.save(outs, 'nets3_minus_n40/%s.pt' % uuid.uuid4())

