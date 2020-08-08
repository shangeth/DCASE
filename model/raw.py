import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Conv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=10, stride=5):
        super(Conv1DBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride) 
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, input):
        out = F.relu(self.bn(self.conv(input)))
        return out


class Residual1DBlock(nn.Module):
    def __init__(self, channels, kernel_size=5, stride=1, padding=2):
        super(Residual1DBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels=channels,
                              out_channels=channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=1) 
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, input):
        residual = input
        out = F.relu(self.bn(self.conv(input)))
        out += residual
        return out

class CNN1d_1s(nn.Module):
    def __init__(self, n_classes, fs, ns):
        super(CNN1d_1s, self).__init__()
        self.fs = fs
        self.ns = ns
        self.sinc_net = nn.Sequential(nn.Conv1d(1, 16, 10, 5),
                                    nn.LeakyReLU(0.3),
                                    nn.BatchNorm1d(16),
                                    nn.MaxPool1d(4, 4),
                                    nn.Conv1d(16, 32, 10, 5),
                                    nn.LeakyReLU(0.3),
                                    nn.BatchNorm1d(32),
                                    nn.MaxPool1d(4, 4),
                                    nn.Conv1d(32, 64, 10, 5),
                                    nn.LeakyReLU(0.3),
                                    nn.BatchNorm1d(64),
                                    nn.MaxPool1d(2, 2)
                     )
        self.avg_pool = nn.AvgPool1d(10, 10, count_include_pad=False)
        self.sinc_net.out_dim = 640
        self.fc_net = nn.Sequential(nn.Linear(self.sinc_net.out_dim, 128),
                            nn.LeakyReLU(0.3),
                            nn.Dropout(0.5),
                            nn.Linear(128, n_classes))

    def forward(self, x):
        bs = x.size(0)
        x = x.view(-1, 1, self.fs)
        x = self.sinc_net(x)
        x = x.transpose(2,0)
        x = self.avg_pool(x).transpose(2,0)

        x = x.reshape(bs, -1)
        x = self.fc_net(x)
        return x
    
    def print_summary(self):
        print('Model Summary')
        summary(self, input_size=(1, self.fs*self.ns))
        print('\n')


class CNN1D(nn.Module):
    def __init__(self, classes_num):
        super(CNN1D, self).__init__()
        self.feature_extractor = nn.Sequential(Conv1DBlock(1, 16),
                                               Conv1DBlock(16, 16),
                                                Residual1DBlock(16),
                                                Conv1DBlock(16, 32),
                                                Conv1DBlock(32, 32),
                                                Residual1DBlock(32),
                                                Conv1DBlock(32, 32))

        self.classifier = nn.Sequential(nn.Linear(1568,1024),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Linear(1024, classes_num))
    def forward(self, input):
        features = self.feature_extractor(input)
        features = features.view(features.size(0), -1)
        out = self.classifier(features)
        return out

if __name__ == "__main__":
    test_input = torch.randn(5, 1, 160000)
    model = CNN1D(3)
    y_hat = model(test_input)
    print(y_hat.shape)