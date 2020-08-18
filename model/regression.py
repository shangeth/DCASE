# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchsummary import summary
# from torchvision import models, transforms

# # class CNN_Regression_1D(nn.Module):
# #     def __init__(self, class_num=1, fs=None, ns=None):
# #         super(CNN_Regression_1D, self).__init__()
# #         self.cnn_network = nn.Sequential(nn.Conv1d(in_channels=26, out_channels=100, kernel_size=5, stride=2),
# #                             nn.ReLU(),
# #                             nn.BatchNorm1d(100),
# #                             nn.Conv1d(in_channels=100, out_channels=50, kernel_size=5, stride=2),
# #                             nn.ReLU(),
# #                             nn.BatchNorm1d(50),
# #                             nn.Conv1d(in_channels=50, out_channels=10, kernel_size=5, stride=2),
# #                             nn.ReLU(),
# #                             nn.BatchNorm1d(10),
# #                             )
# #         self.ann_network = nn.Sequential(nn.Linear(350, 128),
# #                                     nn.ReLU(),
# #                                     nn.Dropout(0.5),
# #                                     nn.Linear(128, class_num))
  
# #     def forward(self, x):
# #         x = x.squeeze(1)
# #         cnn = self.cnn_network(x)
# #         cnn = cnn.view(x.size(0), -1)
# #         out = self.ann_network(cnn)
# #         return out

# #     def print_summary(self):
# #         print('Model Summary')
# #         summary(self, input_size=(26, 307))
# #         print('\n')

# class CNN_Regression_1D(nn.Module):
#     def __init__(self, class_num=1, fs=None, ns=None):
#         super(CNN_Regression_1D, self).__init__()
#         self.cnn_network = nn.Sequential(nn.Conv1d(in_channels=26, out_channels=100, kernel_size=5),
#                             nn.ReLU(),
#                             nn.MaxPool1d(2, 2),
#                             nn.BatchNorm1d(100),
#                             nn.Conv1d(in_channels=100, out_channels=100, kernel_size=5),
#                             nn.ReLU(),
#                             nn.MaxPool1d(2, 2),
#                             nn.BatchNorm1d(100),
#                             nn.Conv1d(in_channels=100, out_channels=100, kernel_size=5),
#                             nn.ReLU(),
#                             nn.MaxPool1d(2, 2),
#                             nn.BatchNorm1d(100),
#                             )
#         # self.ann_network = nn.Sequential(nn.Linear(3400, 128),
#         #                             nn.PReLU(),
#         #                             nn.Dropout(0.5),
#         #                             nn.Linear(128, class_num))
#         self.ann_network = nn.Sequential(nn.Linear(3400, 1)
#                                     )
  
#     def forward(self, x):
#         x = x.squeeze(1)
#         cnn = self.cnn_network(x)
#         cnn = cnn.view(x.size(0), -1)
#         out = self.ann_network(cnn)
#         return out

#     def print_summary(self):
#         print('Model Summary')
#         summary(self, input_size=(26, 307))
#         print('\n')


# class CNN_Regression_2D(nn.Module):
#     def __init__(self, class_num=1, fs=None, ns=None):
#         super(CNN_Regression_2D, self).__init__()
#         self.cnn_network = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 10), stride=(1,2)),
#                             nn.ReLU(),
#                             nn.BatchNorm2d(16),
#                             nn.MaxPool2d(2, 2),
#                             nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 10), stride=(1,2)),
#                             nn.ReLU(),
#                             nn.BatchNorm2d(32),
#                             nn.MaxPool2d(2, 2),
#                             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 6), stride=(1,2)),
#                             nn.ReLU(),
#                             nn.BatchNorm2d(64),
#                             # nn.MaxPool2d(2, 2),
#                             )
        
#         self.ann_network = nn.Sequential(nn.Linear(384, 128),
#                                     nn.ReLU(),
#                                     nn.Dropout(0.5),
#                                     nn.Linear(128, class_num))
  
#     def forward(self, x):
#         # x = x.unsqueeze(1)
#         cnn = self.cnn_network(x)
#         cnn = cnn.view(x.size(0), -1)
#         out = self.ann_network(cnn)
#         return out

#     def print_summary(self):
#         print('Model Summary')
#         summary(self, input_size=(1, 26, 307))
#         print('\n')

# if __name__ == "__main__":
#     model = CNN_Regression_1D()
#     x = torch.randn(5, 1, 26, 307)
#     y_hat = model(x)
#     print(y_hat.shape)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torchvision import models, transforms

# class CNN_Regression_1D(nn.Module):
#     def __init__(self, class_num=1, fs=None, ns=None):
#         super(CNN_Regression_1D, self).__init__()
#         self.cnn_network = nn.Sequential(nn.Conv1d(in_channels=26, out_channels=100, kernel_size=5, stride=2),
#                             nn.ReLU(),
#                             nn.BatchNorm1d(100),
#                             nn.Conv1d(in_channels=100, out_channels=50, kernel_size=5, stride=2),
#                             nn.ReLU(),
#                             nn.BatchNorm1d(50),
#                             nn.Conv1d(in_channels=50, out_channels=10, kernel_size=5, stride=2),
#                             nn.ReLU(),
#                             nn.BatchNorm1d(10),
#                             )
#         self.ann_network = nn.Sequential(nn.Linear(350, 128),
#                                     nn.ReLU(),
#                                     nn.Dropout(0.5),
#                                     nn.Linear(128, class_num))
  
#     def forward(self, x):
#         x = x.squeeze(1)
#         cnn = self.cnn_network(x)
#         cnn = cnn.view(x.size(0), -1)
#         out = self.ann_network(cnn)
#         return out

#     def print_summary(self):
#         print('Model Summary')
#         summary(self, input_size=(26, 307))
#         print('\n')

class CNN_Regression_1D(nn.Module):
    def __init__(self, class_num=1, fs=None, ns=None):
        super(CNN_Regression_1D, self).__init__()
        self.cnn_network = nn.Sequential(nn.Conv1d(in_channels=26, out_channels=100, kernel_size=5),
                            nn.ReLU(),
                            nn.MaxPool1d(2, 2),
                            nn.BatchNorm1d(100),
                            nn.Conv1d(in_channels=100, out_channels=100, kernel_size=5),
                            nn.ReLU(),
                            nn.MaxPool1d(2, 2),
                            nn.BatchNorm1d(100),
                            nn.Conv1d(in_channels=100, out_channels=100, kernel_size=5),
                            nn.ReLU(),
                            nn.MaxPool1d(2, 2),
                            nn.BatchNorm1d(100),
                            nn.Conv1d(in_channels=100, out_channels=100, kernel_size=5),
                            nn.ReLU(),
                            nn.MaxPool1d(2, 2),
                            nn.BatchNorm1d(100),
                            nn.Conv1d(in_channels=100, out_channels=100, kernel_size=5),
                            nn.ReLU(),
                            nn.MaxPool1d(2, 2),
                            nn.BatchNorm1d(100),
                            )
        self.ann_network = nn.Sequential(nn.Linear(1800, 128),
                                    nn.PReLU(),
                                    nn.Dropout(0.5),
                                    nn.Linear(128, class_num))
        # self.ann_network = nn.Sequential(nn.Linear(3400, 1)
        #                             )
  
    def forward(self, x):
        x = x.squeeze(1)
        cnn = self.cnn_network(x)
        cnn = cnn.view(x.size(0), -1)
        out = self.ann_network(cnn)
        return out

    def print_summary(self):
        print('Model Summary')
        summary(self, input_size=(26, 700))
        print('\n')


class CNN_Regression_2D(nn.Module):
    def __init__(self, class_num=1, fs=None, ns=None):
        super(CNN_Regression_2D, self).__init__()
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
                            # nn.MaxPool2d(2, 2),
                            )
        
        self.ann_network = nn.Sequential(nn.Linear(384, 128),
                                    nn.ReLU(),
                                    nn.Dropout(0.5),
                                    nn.Linear(128, class_num))
  
    def forward(self, x):
        # x = x.unsqueeze(1)
        cnn = self.cnn_network(x)
        cnn = cnn.view(x.size(0), -1)
        out = self.ann_network(cnn)
        return out

    def print_summary(self):
        print('Model Summary')
        summary(self, input_size=(1, 26, 307))
        print('\n')

if __name__ == "__main__":
    model = CNN_Regression_1D()
    x = torch.randn(5, 1, 26, 700)
    y_hat = model(x)
    print(y_hat.shape)