import torch
import airbench

def get_loaders(aug=dict(flip=True, translate=2), batch_size=512, **kwargs):
    
    # make normal loader with full CIFAR-10 train set
    loader_all = airbench.CifarLoader('/tmp/cifar10', train=True, aug=aug, batch_size=batch_size, **kwargs)

    indices_random = list(range(20))
    indices_easy = [45114, 47798, 43746, 49106, 47082, 44095, 49524, 41014, 49159, 44279,
                    45927, 40141, 47731, 46440, 49015, 49690, 46836, 43512, 43189, 49320]
    indices = indices_easy+indices_random
    mask = torch.tensor([True]*len(loader_all.images)).cuda()
    mask[indices] = False

    # make a loader which is missing 20 select easy examples, and 20 random examples
    # (for the random examples we just take the first 20 since the set is shuffled)
    loader_minus40 = airbench.CifarLoader('/tmp/cifar10', train=True, aug=aug, batch_size=batch_size, **kwargs)
    loader_minus40.images = loader_minus40.images[mask]
    loader_minus40.labels = loader_minus40.labels[mask]

    return loader_all, loader_minus40
