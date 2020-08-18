# DCASE

## 1. Download Dataset 
Download and extract [DCASE 2019 task 1b dataset](https://zenodo.org/record/2589332), task 1a can also be used without modifying the code.

## 2. Save Features(Raw/mfcc/mel,...)
```
python features/save_features.py
```

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
```
Dataset Statistics
----------
Length of Dataset = 7555
Labels =
Counter({'bus': 1656, 'street_pedestrian': 1656, 'airport': 1656, 'metro': 1221, 'shopping_mall': 1015, 'street_traffic': 351})
Device =
Counter({'a': 6571, 'c': 492, 'b': 492})
Cities =
Counter({'paris': 858, 'helsinki': 852, 'milan': 810, 'lyon': 810, 'barcelona': 798, 'prague': 712, 'london': 707, 'stockholm': 691, 'lisbon': 663, 'vienna': 654})

Data Shape = torch.Size([1, 40, 801])
Model Summary
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 111, 111]           1,792
              ReLU-2         [-1, 64, 111, 111]               0
         MaxPool2d-3           [-1, 64, 55, 55]               0
            Conv2d-4           [-1, 16, 55, 55]           1,040
              ReLU-5           [-1, 16, 55, 55]               0
            Conv2d-6           [-1, 64, 55, 55]           1,088
              ReLU-7           [-1, 64, 55, 55]               0
            Conv2d-8           [-1, 64, 55, 55]           9,280
              ReLU-9           [-1, 64, 55, 55]               0
             Fire-10          [-1, 128, 55, 55]               0
           Conv2d-11           [-1, 16, 55, 55]           2,064
             ReLU-12           [-1, 16, 55, 55]               0
           Conv2d-13           [-1, 64, 55, 55]           1,088
             ReLU-14           [-1, 64, 55, 55]               0
           Conv2d-15           [-1, 64, 55, 55]           9,280
             ReLU-16           [-1, 64, 55, 55]               0
             Fire-17          [-1, 128, 55, 55]               0
        MaxPool2d-18          [-1, 128, 27, 27]               0
           Conv2d-19           [-1, 32, 27, 27]           4,128
             ReLU-20           [-1, 32, 27, 27]               0
           Conv2d-21          [-1, 128, 27, 27]           4,224
             ReLU-22          [-1, 128, 27, 27]               0
           Conv2d-23          [-1, 128, 27, 27]          36,992
             ReLU-24          [-1, 128, 27, 27]               0
             Fire-25          [-1, 256, 27, 27]               0
           Conv2d-26           [-1, 32, 27, 27]           8,224
             ReLU-27           [-1, 32, 27, 27]               0
           Conv2d-28          [-1, 128, 27, 27]           4,224
             ReLU-29          [-1, 128, 27, 27]               0
           Conv2d-30          [-1, 128, 27, 27]          36,992
             ReLU-31          [-1, 128, 27, 27]               0
             Fire-32          [-1, 256, 27, 27]               0
        MaxPool2d-33          [-1, 256, 13, 13]               0
           Conv2d-34           [-1, 48, 13, 13]          12,336
             ReLU-35           [-1, 48, 13, 13]               0
           Conv2d-36          [-1, 192, 13, 13]           9,408
             ReLU-37          [-1, 192, 13, 13]               0
           Conv2d-38          [-1, 192, 13, 13]          83,136
             ReLU-39          [-1, 192, 13, 13]               0
             Fire-40          [-1, 384, 13, 13]               0
           Conv2d-41           [-1, 48, 13, 13]          18,480
             ReLU-42           [-1, 48, 13, 13]               0
           Conv2d-43          [-1, 192, 13, 13]           9,408
             ReLU-44          [-1, 192, 13, 13]               0
           Conv2d-45          [-1, 192, 13, 13]          83,136
             ReLU-46          [-1, 192, 13, 13]               0
             Fire-47          [-1, 384, 13, 13]               0
           Conv2d-48           [-1, 64, 13, 13]          24,640
             ReLU-49           [-1, 64, 13, 13]               0
           Conv2d-50          [-1, 256, 13, 13]          16,640
             ReLU-51          [-1, 256, 13, 13]               0
           Conv2d-52          [-1, 256, 13, 13]         147,712
             ReLU-53          [-1, 256, 13, 13]               0
             Fire-54          [-1, 512, 13, 13]               0
           Conv2d-55           [-1, 64, 13, 13]          32,832
             ReLU-56           [-1, 64, 13, 13]               0
           Conv2d-57          [-1, 256, 13, 13]          16,640
             ReLU-58          [-1, 256, 13, 13]               0
           Conv2d-59          [-1, 256, 13, 13]         147,712
             ReLU-60          [-1, 256, 13, 13]               0
             Fire-61          [-1, 512, 13, 13]               0
          Dropout-62          [-1, 512, 13, 13]               0
           Conv2d-63            [-1, 6, 13, 13]           3,078
             ReLU-64            [-1, 6, 13, 13]               0
AdaptiveAvgPool2d-65              [-1, 6, 1, 1]               0
       SqueezeNet-66                    [-1, 6]               0
================================================================
Total params: 725,574
Trainable params: 605,158
Non-trainable params: 120,416
----------------------------------------------------------------
Input size (MB): 0.57
Forward/backward pass size (MB): 51.19
Params size (MB): 2.77
Estimated Total Size (MB): 54.53
----------------------------------------------------------------


  9%|▉         | 5/54 [00:14<02:26,  2.99s/it]
```

## 4. Test 

## 5. Inference



# Results
| Feature | Model            | Model Size    | Training Time (200) Epoch | Acc               | Precision | Recall | f1   |
|---------|------------------|---------------|---------------------------|-------------------|-----------|--------|------|
| Raw     | 1DCNN            |               |                           |                   |           |        |      |
|         | 1DCNN_1s         |               |                           |                   |           |        |      |
|         | CNN-RNN          |               |                           |                   |           |        |      |
|         | SincNet          |               |                           |                   |           |        |      |
|         | PASE             |               |                           |                   |           |        |      |
|         | PASE(Pretrained) |               |                           |                   |           |        |      |
|         |                  |               |                           |                   |           |        |      |
| MFCC    | 1DCNN            | 191670 / 7.73 |                           | 0.8582            | 0.86      | 0.86   | 0.86 |
|         | 2DCNN            |               |                           | 0.8556            | 0.86      | 0.86   | 0.85 |
|         | CNN-RNN          |               |                           |                   |           |        |      |
|         | Pretrained CNN   |               |                           |                   |           |        |      |
|         |                  |               |                           |                   |           |        |      |
| Mel     | 1DCNN            | 96350 / 0.69  |                           | 0.6450(86 epochs) |           |        |      |
|         | 2DCNN            |               |                           | 0.7682            | 0.78      | 0.78   | 0.78 |
|         | CNN-RNN          |               |                           |                   |           |        |      |
|         | Pretrained CNN   |               |                           |                   |           |        |      |
