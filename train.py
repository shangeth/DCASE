from dataset.Dataset20191b import RawWaveDataset
from dataset.utils import get_dataloader
from models.raw import CNN1D
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DATA_DIR = 'data/audio'
dataset = RawWaveDataset(DATA_DIR)
dataset.print_stats()
trainloader, valloader = get_dataloader(dataset)


model = CNN1D(dataset.class_num)
