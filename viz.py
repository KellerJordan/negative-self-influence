import sys
import torch
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import scipy.stats
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
indices = list(range(20))+indices

i = 2
if len(sys.argv) == 2:
    i = int(sys.argv[1])
print('Loading the outputs produced by all the models that were trained...')
outs1 = torch.stack([torch.load(p) for p in tqdm(glob.glob('nets%d_default/*' % i))])
outs2 = torch.stack([torch.load(p) for p in tqdm(glob.glob('nets%d_minus_n40/*' % i))])

print('Computing the correct-class margins for each example which was ablated')
mm1 = margin(outs1, train_loader.labels).float().cpu()
mm2 = margin(outs2, train_loader.labels).float().cpu()
diff = (mm1.mean(0) - mm2.mean(0))
std = (mm1.var(0) / len(mm1) + mm2.var(0) / len(mm2))**0.5
zz = (diff / std) # z-score of each difference

print('Example index\t   margin  \t\tself-influence\t\t\tp-value')
print('\t\twith\twithout')
print()
print('Random examples:')
for j in range(40):
    i = indices[j]
    z_score = zz[i].item()
    p_value = scipy.stats.norm.sf(abs(z_score)) * 2
    print('%d\t\t%.3f\t%.3f\t\t%+.3f\t\t\t\t%.4f' % (i, mm2.mean(0)[i], mm1.mean(0)[i], diff[i], p_value))
    if j == 19:
        print('Average:\t\t\t\t%+.3f' % diff[indices[:20]].mean())
        print()
        print('Easy examples:')
    if j == 39:
        print('Average:\t\t\t\t%+.3f' % diff[indices[20:]].mean())

