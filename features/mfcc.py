from tqdm import tqdm
import librosa
import os
import numpy as np
import torchaudio
import torch

def get_mfcc(DATA_PATH, SAVE_DIR, samplerate=16000):
    audio_list = os.listdir(DATA_PATH)
    for name in tqdm(audio_list):
        arr_name = name.split('.')[0]
        waveform, sample_rate = librosa.load(DATA_PATH+'/'+name, sr=samplerate)
        arr = torchaudio.transforms.MFCC(sample_rate=sample_rate)(torch.tensor(waveform)).numpy()
        np.save(SAVE_DIR+'/'+arr_name+'.npy', arr)

def get_mel_spectrogram(DATA_PATH, SAVE_DIR, samplerate=16000, n_mels=40):
    audio_list = os.listdir(DATA_PATH)
    for name in tqdm(audio_list):
        arr_name = name.split('.')[0]
        waveform, sample_rate = librosa.load(DATA_PATH+'/'+name, sr=samplerate)
        feature = librosa.feature.melspectrogram(waveform,
                                sr=samplerate,
                                n_fft=int(0.04*samplerate),
                                hop_length=int(0.02*samplerate),
                                n_mels=n_mels)
        feature = np.log10(np.maximum(feature, 1e-10))
        np.save(SAVE_DIR+'/'+arr_name+'.npy', feature)

if __name__ == "__main__":
    DATA_PATH = 'data/audio'
    SAVE_DIR = 'data/features'
    get_mfcc(DATA_PATH, SAVE_DIR)