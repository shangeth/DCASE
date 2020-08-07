import librosa
import torchaudio
from torch.utils.data import Dataset
import os
from fnmatch import fnmatch
import torch
from collections import Counter

class RawWaveDataset(Dataset):
    def __init__(self, root_dir, test=False, sampling_rate=16000):
        self.root_dir = root_dir
        self.test = test
        self.wav_files, self.label_counter, self.device_counter, self.city_counter = self.get_wav_files(self.root_dir)

        self.labels_list =sorted(list(self.label_counter.keys()))
        self.device_list = sorted(list(self.device_counter.keys()))
        self.sampling_rate = sampling_rate
        
    def __len__(self):
        return len(self.wave_files)

    def get_wav_files(self, root, pattern = "*.wav"):
        wav_files = []
        labels = []
        devices = []
        cities = []
        for path, subdirs, files in os.walk(root):
            for name in files:
                if fnmatch(name, pattern):
                    wav_path = os.path.join(path, name)
                    if self.test: 
                        label = None
                        device = None
                        city = None
                    else: 
                        label = name.split('-')[0]
                        device = name.split('-')[-1].split('.')[0]
                        city = name.split('-')[1]
                    wav_files.append((wav_path, label, device))
                    labels.append(label)
                    devices.append(device)
                    cities.append(city)
        label_counter = Counter(labels)
        device_counter = Counter(devices)
        city_counter = Counter(cities)
        return wav_files, label_counter, device_counter, city_counter

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name, label, device = self.wave_files[idx]
        waveform, sample_rate = torchaudio.load(file_name)
        waveform = torchaudio.transforms.Resample(sample_rate, self.sampling_rate)(waveform.view(1,-1))
        waveform = waveform[:, :10*self.sampling_rate]
        return waveform, self.labels_list.index(label), self.device_list.index(device)




if __name__ == "__main__":

    DATA_PATH = 'data/audio'
    dataset = RawWaveDataset(DATA_PATH)
    print(dataset[0][0].shape)