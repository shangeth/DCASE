# DCASE

## 1. Download Dataset 
## 2. Change config.py
```
python model/model_lookup.py
```
```
{'mel': {'1d': <class 'spectral.CNN_MEL_1D'>,
         '2d': <class 'spectral.CNN_MEL_2D'>,
         'squeezenet': <class 'spectral.SqueezeNet'>,
         'vgg': <class 'spectral.VGG'>},
 'mfcc': {'1d': <class 'spectral.CNN_MFCC_2D'>,
          '2d': <class 'spectral.CNN_MFCC_2D'>,
          'squeezenet': <class 'spectral.SqueezeNet'>,
          'vgg': <class 'spectral.VGG'>},
 'raw': {'1d': <class 'raw.CNN1d_1s'>, '2d': <class 'raw.CNN1D'>}}
```
```
python config.py
```
```
{'audio_features': 'mfcc',
 'audio_sample_rate': 16000,
 'augment': True,
 'batch_size': 128,
 'data_dir': 'data/mfcc',
 'epochs': 10,
 'log_path': 'logging/',
 'lr': 0.001,
 'model_type': 'vgg',
 'pretrained': True,
 'save_model_file': 'trained_model.pt',
 'tensorboard_path': 'logging/runs/',
 'val_split_ratio': 0.1}
```
## 3. Train the model
```
python train.py
```
