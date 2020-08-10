import torch.utils.data as data
from dataset.Dataset20191b import ApplySpectralAug, ApplyRawAug, ApplySpectralPretrainedAug

def random_split(dataset, val_ratio=0.1):
    val_len = int(len(dataset)*val_ratio)
    train_len = len(dataset) - val_len
    train_set, val_set = data.random_split(dataset, [train_len, val_len])
    return train_set, val_set

def get_dataloader(dataset, train_bs=50, valid_bs=50, val_ratio=0.1, augment=True, spectral=True, pretrained=True):
    train_set, val_set = random_split(dataset, val_ratio)
    if spectral and augment:
        train_set = ApplySpectralAug(train_set)
    if pretrained:
        train_set = ApplySpectralPretrainedAug(train_set)
        val_set = ApplySpectralPretrainedAug(val_set)
    if augment and not spectral:
        train_set = ApplyRawAug(train_set)
    trainloader = data.DataLoader(train_set, batch_size=train_bs, shuffle=True)
    valloader = data.DataLoader(val_set, batch_size=valid_bs, shuffle=False)    
    return trainloader, valloader