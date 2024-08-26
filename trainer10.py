import airbench

def train(loader):
    return airbench.train94(loader, label_smoothing=0, learning_rate=5.0)

import os
from tqdm import tqdm
import uuid
import torch
import airbench
from utils import get_loaders

if __name__ == '__main__':

    loader_all, loader_minus40 = get_loaders(batch_size=1000)

    os.makedirs('nets10_default', exist_ok=True)
    for _ in tqdm(range(1000)):
        net = train(loader_all)
        outs = airbench.infer(net, loader_all)
        torch.save(outs, 'nets10_default/%s.pt' % uuid.uuid4())

    os.makedirs('nets10_minus_n40', exist_ok=True)
    for _ in tqdm(range(1000)):
        net = train(loader_minus40)
        outs = airbench.infer(net, loader_all)
        torch.save(outs, 'nets10_minus_n40/%s.pt' % uuid.uuid4())

