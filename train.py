from dataset.Dataset20191b import RawWaveDataset
from dataset.utils import get_dataloader
from models.raw import CNN1d_1s
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DATA_DIR = 'data/audio'
dataset = RawWaveDataset(DATA_DIR)
dataset.print_stats()
trainloader, valloader = get_dataloader(dataset)


model = CNN1d_1s(dataset.class_num)
model.to(device)
print(model)

test_x = torch.randn(1, 1, 10*44100)
y_test = model(test_x.to(device))
print(y_test.shape)