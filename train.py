from dataset.Dataset20191b import RawWaveDataset


DATA_DIR = 'data/audio'
dataset = RawWaveDataset(DATA_DIR)
print(dataset[0][0])