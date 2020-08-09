import librosa
import torchaudio
from torch.utils.data import Dataset
import os
from fnmatch import fnmatch
import torch
from collections import Counter
import numpy
import random
from dataset.data_aug import time_mask, time_warp, freq_mask
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# https://pypi.org/project/audiomentations/
# https://github.com/makcedward/nlpaug
class RawWaveDataset(Dataset):
    def __init__(self, root_dir, test=False, undersample=False, sampling_rate=16000):
        self.root_dir = root_dir
        self.test = test
        self.wav_files, self.label_counter, self.device_counter, self.city_counter = self.get_wav_files(self.root_dir)

        self.labels_list =sorted(list(self.label_counter.keys()))
        self.device_list = sorted(list(self.device_counter.keys()))
        self.undersample = undersample
        self.sampling_rate = sampling_rate
        self.class_num = len(self.labels_list)
        if self.undersample:
            self.fs = self.sampling_rate
        else: 
            self.fs = 44100
        self.ns = 10
        
    def __len__(self):
        return len(self.wav_files)

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

        file_name, label, device = self.wav_files[idx]
        waveform, sample_rate = torchaudio.load(file_name)
        if self.undersample:
            waveform = torchaudio.transforms.Resample(sample_rate, self.sampling_rate)(waveform)
        waveform = waveform[:, :self.ns*self.fs]
        return waveform, self.labels_list.index(label), self.device_list.index(device)

    def print_stats(self):
        print('\nDataset Statistics')
        print('-'*10)
        print(f'Length of Dataset = {len(self.wav_files)}')
        print(f'Labels =\n{self.label_counter}')
        print(f'Device =\n{self.device_counter}')
        print(f'Cities =\n{self.city_counter}\n')

class MFCC_Dataset(Dataset):
    def __init__(self, root_dir, test=False, undersample=False, sampling_rate=16000, dim2d=True):
        self.root_dir = root_dir
        self.test = test
        self.dim2d = dim2d
        self.wav_files, self.label_counter, self.device_counter, self.city_counter = self.get_wav_files(self.root_dir)

        self.labels_list =sorted(list(self.label_counter.keys()))
        self.device_list = sorted(list(self.device_counter.keys()))
        self.class_num = len(self.labels_list)
        if undersample:
            self.fs = sampling_rate
        else: 
            self.fs = 44100
        self.ns = 10
        
    def __len__(self):
        return len(self.wav_files)

    def get_wav_files(self, root, pattern = "*.npy"):
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

        file_name, label, device = self.wav_files[idx]
        waveform = torch.tensor(numpy.load(file_name)).float()
    
        waveform = waveform.unsqueeze(0)
        if random.random() >= 0.5:
            waveform = time_mask(waveform, num_masks=2)
        
        if random.random() >= 0.5:
            waveform = freq_mask(waveform, num_masks=2)

        if random.random() >= 0.5:
            waveform = time_warp(waveform)

        if not self.dim2d:
            waveform = waveform.squeeze(0)
        return waveform, self.labels_list.index(label), self.device_list.index(device)

    def print_stats(self):
        print('\nDataset Statistics')
        print('-'*10)
        print(f'Length of Dataset = {len(self.wav_files)}')
        print(f'Labels =\n{self.label_counter}')
        print(f'Device =\n{self.device_counter}')
        print(f'Cities =\n{self.city_counter}\n')


if __name__ == "__main__":

    DATA_PATH = 'data/audio'
    dataset = RawWaveDataset(DATA_PATH)
    print(dataset[0][0].shape)