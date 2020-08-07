import librosa
import torchaudio
from torch.utils.data import Dataset
import os
from fnmatch import fnmatch
import torch

class RawWaveDataset(Dataset):
    def __init__(self, root_dir, labels_list, device_list, test=False, sampling_rate=16000):
        self.root_dir = root_dir
        self.test = test
        self.wave_files = self.get_wav_files(self.root_dir)
        self.labels_list = labels_list 
        self.device_list = device_list
        self.sampling_rate = sampling_rate
        
    def __len__(self):
        return len(self.wave_files)

    def get_wav_files(self, root, pattern = "*.wav"):
        wav_files = []
        for path, subdirs, files in os.walk(root):
            for name in files:
                if fnmatch(name, pattern):
                    wav_path = os.path.join(path, name)
                    if self.test: 
                        label = None
                        device = None
                    else: 
                        label = name.split('-')[0]
                        device = name.split('-')[-1].split('.')[0]
                    wav_files.append((wav_path, label, device))
        return wav_files

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
    audio_list = os.listdir(DATA_PATH)

    for name in audio_list:
        label = name.split('-')[0]
        device = name.split('-')[-1].split('.')[0]
        labels.append(label)
        devices.append(device)

    labels_set =sorted(list(set(labels)))
    devices_set = sorted(list(set(devices)))


    dataset = RawWaveDataset(DATA_PATH, labels_set, devices_set)
    print(dataset[0][0].shape)