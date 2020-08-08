import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class CNN_MFCC(nn.Module):
    def __init__(self, class_num):
        super(DCASE, self).__init__()
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