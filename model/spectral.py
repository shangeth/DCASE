import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torchvision import models, transforms

class CNN_MEL_1D(nn.Module):
    def __init__(self, class_num):
        super(CNN_MEL_1D, self).__init__()
        self.cnn_network = nn.Sequential(nn.Conv1d(in_channels=40, out_channels=100, kernel_size=10, stride=5),
                            nn.ReLU(),
                            nn.BatchNorm1d(100),
                            nn.Conv1d(in_channels=100, out_channels=50, kernel_size=10, stride=5),
                            nn.ReLU(),
                            nn.BatchNorm1d(50),
                            nn.Conv1d(in_channels=50, out_channels=10, kernel_size=10, stride=5),
                            nn.ReLU(),
                            nn.BatchNorm1d(10),
                            )
        self.ann_network = nn.Sequential(nn.Linear(50, 32),
                                    nn.ReLU(),
                                    nn.Dropout(0.5),
                                    nn.Linear(32, class_num))
  
    def forward(self, x):
        cnn = self.cnn_network(x)
        cnn = cnn.view(x.size(0), -1)
        out = self.ann_network(cnn)
        return out

    def print_summary(self):
        print('Model Summary')
        summary(self, input_size=(1, 40, 801))
        print('\n')

class CNN_MEL_2D(nn.Module):
    def __init__(self, class_num, fs, ns):
        super(CNN_MEL_2D, self).__init__()
        self.cnn_network = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 10), stride=(1,2)),
                            nn.ReLU(),
                            nn.BatchNorm2d(16),
                            nn.MaxPool2d(2, 2),
                            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 10), stride=(1,2)),
                            nn.ReLU(),
                            nn.BatchNorm2d(32),
                            nn.MaxPool2d(2, 2),
                            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 6), stride=(1,2)),
                            nn.ReLU(),
                            nn.BatchNorm2d(64),
                            nn.MaxPool2d(2, 2),
                            )
        
        self.ann_network = nn.Sequential(nn.Linear(768, 128),
                                    nn.ReLU(),
                                    nn.Dropout(0.5),
                                    nn.Linear(128, class_num))
  
    def forward(self, x):
        cnn = self.cnn_network(x)
        cnn = cnn.view(x.size(0), -1)
        out = self.ann_network(cnn)
        return out

    def print_summary(self):
        print('Model Summary')
        summary(self, input_size=(1, 40, 501))
        print('\n')

class CNN_MFCC_2D(nn.Module):
    def __init__(self, class_num, fs, ns):
        super(CNN_MFCC_2D, self).__init__()
        self.cnn_network = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 10), stride=(1,2)),
                            nn.ReLU(),
                            nn.BatchNorm2d(16),
                            nn.MaxPool2d(2, 2),
                            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 10), stride=(1,2)),
                            nn.ReLU(),
                            nn.BatchNorm2d(32),
                            nn.MaxPool2d(2, 2),
                            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 10), stride=(1,2)),
                            nn.ReLU(),
                            nn.BatchNorm2d(64),
                            nn.MaxPool2d(2, 2),
                            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 5), stride=(1,1)),
                            nn.ReLU(),
                            nn.BatchNorm2d(64),
                            )
        
        self.ann_network = nn.Sequential(nn.Linear(640, 128),
                                    nn.ReLU(),
                                    nn.Dropout(0.5),
                                    nn.Linear(128, class_num))
  
    def forward(self, x):
        cnn = self.cnn_network(x)
        cnn = cnn.view(x.size(0), -1)
        out = self.ann_network(cnn)
        return out

    def print_summary(self):
        print('Model Summary')
        summary(self, input_size=(1, 40, 801))
        print('\n')


class VGG(nn.Module):
    def __init__(self, class_num, fs, ns):
        super(VGG, self).__init__()
        self.vgg = models.vgg16(pretrained=False)
        
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.vgg.classifier[6] = nn.Sequential(
                            nn.Linear(4096, 256), 
                            nn.ReLU(), 
                            nn.Dropout(0.4),
                            nn.Linear(256, class_num))
  
    def forward(self, x):
        out = self.vgg(x)
        return out

    def print_summary(self):
        print('Model Summary')
        summary(self, input_size=(3, 224, 224))
        print('\n')