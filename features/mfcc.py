from tqdm import tqdm
import librosa
import os
import numpy as np
import torchaudio
import torch

def get_mfcc(DATA_PATH, SAVE_DIR, resample=16000):
    audio_list = os.listdir(DATA_PATH, SAVE_DIR)
    for name in tqdm(audio_list):
        arr_name = name.split('.')[0]
        waveform, sample_rate = librosa.load(DATA_PATH+'/'+name, sr=16000)
        arr = torchaudio.transforms.MFCC(sample_rate=sample_rate)(torch.tensor(waveform)).numpy()
        np.save(SAVE_DIR+'/'+arr_name+'.npy', arr)

if __name__ == "__main__":
    DATA_PATH = ''
    SAVE_DIR = ''
    get_mfcc(DATA_PATH, SAVE_DIR)