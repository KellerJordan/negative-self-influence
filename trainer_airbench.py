import os
from tqdm import tqdm
import uuid
import torch
import airbench
from utils import get_loaders

def train(loader):
    return airbench.train94(loader, label_smoothing=0, learning_rate=5.0)

if __name__ == '__main__':

    loader_all, loader_minus40 = get_loaders()

    os.makedirs('nets2_default', exist_ok=True)
    for _ in tqdm(range(100)):
        net = train(loader_all)
        outs = airbench.infer(net, loader_all)
        torch.save(outs, 'nets2_default/%s.pt' % uuid.uuid4())

    os.makedirs('nets2_minus_n40', exist_ok=True)
    for _ in tqdm(range(100)):
        net = train(loader_minus40)
        outs = airbench.infer(net, loader_all)
        torch.save(outs, 'nets2_minus_n40/%s.pt' % uuid.uuid4())
