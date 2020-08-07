from dataset.Dataset20191b import RawWaveDataset
from dataset.utils import get_dataloader
from models.raw import CNN1d_1s
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DATA_DIR = 'data/audio'
dataset = RawWaveDataset(DATA_DIR)
dataset.print_stats()
fs = dataset.fs
ns = dataset.ns
trainloader, valloader = get_dataloader(dataset)


model = CNN1d_1s(dataset.class_num, fs, ns)
model.to(device)
model.print_summary()

test_x = torch.randn(1, 1, fs*ns)
y_test = model(test_x.to(device))
print(y_test.shape)