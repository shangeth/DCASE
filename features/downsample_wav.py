import librosa
import os
from tqdm import tqdm

def downsample_wav_files(DATA_PATH, SAVE_DIR, samplerate=16000):
    audio_list = os.listdir(DATA_PATH)
    for name in tqdm(audio_list): 
        y, sr = librosa.load(DATA_PATH+'/'+name, sr=samplerate)
        librosa.output.write_wav(SAVE_DIR + '/' + name, y, sr)

if __name__ == "__main__":
    DATA_PATH = 'data/audio'
    SAVE_DIR = 'data/audio_16k'
    downsample_wav_files(DATA_PATH, SAVE_DIR)
