from torch.utils.data import Dataset
import os
import torch
import random
import numpy as np
from dataset.data_aug import time_mask, time_warp, freq_mask

class Timit_Dataset(Dataset):
    def __init__(self, root_dir, train=True, l=700, label_range=(144.78, 203.2)):
        self.root_dir = root_dir
        self.train = train
        self.files = os.listdir(self.root_dir)
        self.l = l
        self.label_range = label_range
        
        
    def __len__(self):
        return len(self.files)

    def pad_crop_features(self, arr, l, train):
        t_len = arr.shape[1]
        for _ in range(5):
            if t_len <= (int(l/2)):
                arr = np.concatenate([arr, arr], 1)
                t_len = arr.shape[1]
            else: break

        if t_len > l:
            if train: i = np.random.randint(0, t_len-l)
            else: i=0
            arr = arr[:, i : i+l]
            return arr
        
        elif t_len < l:
            arr_new = np.random.randn(arr.shape[0], l)
            arr_new[:, :t_len] = arr
            
            delta = l-t_len
            if train: 
                i = np.random.randint(0, t_len-delta)

            else: i = 0
            arr_new[:, t_len:] = 0
            # arr[:, i:i+delta]
            return arr_new

        else:
            return arr

    def normalize_label(self, label):
        m, M = self.label_range
        label = (label - m)/(M - m)
        return label

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        file = self.files[idx]
        waveform = np.load(self.root_dir + '/' + file)
        label = float('.'.join(file.split('_')[-1].split('.')[:-1]))

        waveform = self.pad_crop_features(waveform, self.l, self.train)
        waveform = waveform/6.0
        waveform = torch.from_numpy(waveform).unsqueeze(0)
        label = self.normalize_label(label)
        
        if self.train:
            # if random.random() >= 0.3:
            waveform = time_mask(waveform, num_masks=2)
                
            # if random.random() >= 0.3:
            waveform = freq_mask(waveform, num_masks=2)

            # if random.random() >= 0.5:
            #     waveform = time_warp(waveform)

        return waveform, label

    def print_stats(self):
        print('\nDataset Statistics')
        print('-'*10)
        print(f'Length of Dataset = {len(self.files)}')

