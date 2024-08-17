import torch
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import airbench

def margin(outs, labs):
    ind = torch.arange(len(labs), device=labs.device)
    tgts = outs[..., ind, labs]
    outs_other = outs.clone()
    outs_other[..., ind, labs] = -torch.inf
    return tgts - outs_other.logsumexp(-1)

train_loader = airbench.CifarLoader('/tmp/cifar10', train=True)
indices = [45114, 47798, 43746, 49106, 47082, 44095, 49524, 41014, 49159, 44279,
           45927, 40141, 47731, 46440, 49015, 49690, 46836, 43512, 43189, 49320]
indices = list(range(30))+indices

print('Loading the outputs produced by all the models that were trained...')
outs1 = torch.stack([torch.load(p) for p in tqdm(glob.glob('nets1_default/*'))])
outs2 = torch.stack([torch.load(p) for p in tqdm(glob.glob('nets1_minus_n40/*'))])

print('Computing the correct-class margins for each example which was ablated')
mm1 = margin(outs1, train_loader.labels)
mm2 = margin(outs2, train_loader.labels)

print('The impact of *including* the example on its own correct-class margin was as follows...')
diff = (mm1.mean(0) - mm2.mean(0))
diff1 = diff[indices[:20]]
diff2 = diff[indices[20:]]
print('Random examples:', diff1)
print('Easy examples:', diff2)
print('Random mean:', diff1.mean())
print('Easy mean:', diff2.mean())

