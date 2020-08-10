from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import numpy as np
from tqdm import tqdm
SAMPLE_RATE = 44100

augmenter = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
])

samples = np.random.randn(100, SAMPLE_RATE*10)
for i in tqdm(range(samples.shape[0])):
    print(augmenter(samples=samples[i], sample_rate=SAMPLE_RATE).shape)