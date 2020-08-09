import torch.utils.data as data
from dataset.Dataset20191b import ApplyAug


def random_split(dataset, val_ratio=0.1):
    val_len = int(len(dataset)*val_ratio)
    train_len = len(dataset) - val_len
    train_set, val_set = data.random_split(dataset, [train_len, val_len])
    return train_set, val_set

def get_dataloader(dataset, train_bs=50, valid_bs=50):
    train_set, val_set = random_split(dataset)
    train_set = ApplyAug(train_set)
    trainloader = data.DataLoader(train_set, batch_size=train_bs, shuffle=True)
    valloader = data.DataLoader(val_set, batch_size=valid_bs, shuffle=False)    
    return trainloader, valloader