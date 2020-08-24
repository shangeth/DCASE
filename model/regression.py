
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torchvision import models, transforms

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
#                             nn.Conv1d(in_channels=100, out_channels=100, kernel_size=5),
#                             nn.ReLU(),
#                             nn.MaxPool1d(2, 2),
#                             nn.BatchNorm1d(100),
#                             nn.Conv1d(in_channels=100, out_channels=100, kernel_size=5),
#                             nn.ReLU(),
#                             nn.MaxPool1d(2, 2),
#                             nn.BatchNorm1d(100),
#                             )
#         self.ann_network = nn.Sequential(nn.Linear(500, 128),
#                                     nn.PReLU(),
#                                     nn.Dropout(0.5),
#                                     nn.Linear(128, class_num))
#         # self.ann_network = nn.Sequential(nn.Linear(3400, 1)
#         #                             )
  
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
                            # nn.Conv1d(in_channels=100, out_channels=100, kernel_size=5),
                            # nn.ReLU(),
                            # nn.MaxPool1d(2, 2),
                            # nn.BatchNorm1d(100),
                            )
        # self.ann_network = nn.Sequential(nn.Linear(800, 64),
        #                             nn.PReLU(),
        #                             nn.Dropout(0.5),
        #                             nn.Linear(64, class_num))
        self.ann_network = nn.Sequential(nn.Linear(800, 1))
  
    def forward(self, x):
        x = x.squeeze(1)
        cnn = self.cnn_network(x)
        cnn = cnn.view(x.size(0), -1)
        out = self.ann_network(cnn)
        return out

    def print_summary(self):
        print('Model Summary')
        summary(self, input_size=(26, 200))
        print('\n')

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

# 700
# class CNN_Regression_1D(nn.Module):
#     def __init__(self, class_num=1, fs=None, ns=None):
#         super(CNN_Regression_1D, self).__init__()
#         self.cnn_network = nn.Sequential(nn.Conv1d(in_channels=26, out_channels=32, kernel_size=10),
#                             nn.ReLU(),
#                             nn.MaxPool1d(6, 3),
#                             nn.BatchNorm1d(32),
#                             nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5),
#                             nn.ReLU(),
#                             nn.MaxPool1d(6, 3),
#                             nn.BatchNorm1d(64),
#                             nn.Conv1d(in_channels=64, out_channels=100, kernel_size=5),
#                             nn.ReLU(),
#                             nn.MaxPool1d(4, 2),
#                             nn.BatchNorm1d(100),
#                             nn.Conv1d(in_channels=100, out_channels=100, kernel_size=5),
#                             nn.ReLU(),
#                             nn.MaxPool1d(4, 2),
#                             nn.BatchNorm1d(100),
#                             nn.Conv1d(in_channels=100, out_channels=100, kernel_size=5),
#                             nn.ReLU(),
#                             nn.MaxPool1d(4, 2),
#                             nn.BatchNorm1d(100),
#                             )
#         self.ann_network = nn.Sequential(nn.Linear(400, 128),
#                                     nn.PReLU(),
#                                     nn.Dropout(0.5),
#                                     nn.Linear(128, class_num))
#         # self.ann_network = nn.Sequential(nn.Linear(3400, 1)
#         #                             )
  
#     def forward(self, x):
#         x = x.squeeze(1)
#         cnn = self.cnn_network(x)
#         cnn = cnn.view(x.size(0), -1)
#         out = self.ann_network(cnn)
#         return out

#     def print_summary(self):
#         print('Model Summary')
#         summary(self, input_size=(26, 700))
#         print('\n')


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
        
        self.ann_network = nn.Sequential(nn.Linear(384, class_num))
                                    # nn.ReLU(),
                                    # nn.Dropout(0.5),
                                    # nn.Linear(128, class_num))
  
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


class CNN_Regression_2D_700(nn.Module):
    def __init__(self, class_num=1, fs=None, ns=None):
        super(CNN_Regression_2D_700, self).__init__()
        self.cnn_network = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 10), stride=(1,2)),
                            nn.ReLU(),
                            nn.BatchNorm2d(16),
                            nn.MaxPool2d(2, (1,2)),
                            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 10), stride=(1,2)),
                            nn.ReLU(),
                            nn.BatchNorm2d(32),
                            nn.MaxPool2d(2, (1,2)),
                            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 6), stride=(1,2)),
                            nn.ReLU(),
                            nn.BatchNorm2d(64),
                            nn.MaxPool2d(2, (1,2)),
                            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 6), stride=(1,2)),
                            nn.ReLU(),
                            nn.BatchNorm2d(64),
                            nn.MaxPool2d(2, (1,2)),
                            )
        
        self.ann_network = nn.Sequential(nn.Linear(640, class_num))
                                    # nn.ReLU(),
                                    # nn.Dropout(0.5),
                                    # nn.Linear(128, class_num))
  
    def forward(self, x):
        # x = x.unsqueeze(1)
        cnn = self.cnn_network(x)
        cnn = cnn.view(x.size(0), -1)
        out = self.ann_network(cnn)
        return out

    def print_summary(self):
        print('Model Summary')
        summary(self, input_size=(1, 26, 700))
        print('\n')


class LSTM_Regression(nn.Module):
    def __init__(self, class_num=1, fs=None, ns=None):
        super(LSTM_Regression, self).__init__()
        input_size=26
        time_len=700 
        self.n_hidden = 64
        self.n_layers = 1
        self.time_len = ns
        self.input_size = input_size
        self.l_lstm = nn.LSTM(input_size = self.input_size, 
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers, 
                                 batch_first = True)
        self.avg_pool = nn.AvgPool1d(700)
        self.ann_network = nn.Sequential(
                                    nn.Linear(self.n_hidden, 32),
                                    nn.ReLU(),
                                    nn.Dropout(0.5),
                                    nn.Linear(32, class_num))
    def init_hidden(self, batch_size):
        hidden_state = torch.zeros(self.n_layers,batch_size,self.n_hidden)
        cell_state = torch.zeros(self.n_layers,batch_size,self.n_hidden)
        self.hidden = (hidden_state, cell_state)

    def forward(self, x):
        bs = x.size(0)
        self.hidden = self.init_hidden(bs)
        x = x.squeeze(1)
        x = x.view(bs, self.time_len, self.input_size)
        lstm_out = self.l_lstm(x ,self.hidden)
        out, h = lstm_out
        # out = out[:, -1, :]
        out = torch.mean(out, 1)
        out = out.view(bs, -1)
        out = self.ann_network(out)
        return out

    def print_summary(self):
        print('Model Summary')
        summary(self, input_size=(1, self.input_size, self.time_len))
        print('\n')

class CNN_LSTM_Regression(nn.Module):
    def __init__(self, class_num=1, fs=None, ns=None):
        super(CNN_LSTM_Regression, self).__init__()
        input_size=26
        
        self.n_layers = 1
        self.time_len = ns
        self.input_size = 50
        self.n_hidden = 25

        self.cnn_network = self.cnn_network = nn.Sequential(nn.Conv1d(in_channels=26, out_channels=int(self.input_size/2), kernel_size=5),
                            nn.ReLU(),
                            nn.MaxPool1d(4, 2),
                            nn.BatchNorm1d(int(self.input_size/2)),
                            # nn.Conv1d(in_channels=int(self.input_size/2), out_channels=self.input_size, kernel_size=5),
                            # nn.ReLU(),
                            # nn.MaxPool1d(4, 2),
                            # nn.BatchNorm1d(self.input_size),
                            # nn.Conv1d(in_channels=self.input_size, out_channels=self.input_size, kernel_size=5),
                            # nn.ReLU(),
                            # nn.MaxPool1d(4, 2),
                            # nn.BatchNorm1d(self.input_size)
                            )
        self.l_lstm = nn.LSTM(input_size = int(self.input_size/2), 
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers, 
                                 batch_first = True)
        self.avg_pool = nn.AvgPool1d(700)
        self.ann_network = nn.Sequential(
                                    nn.Linear(self.n_hidden, class_num))
                                    # nn.ReLU(),
                                    # nn.Dropout(0.5),
                                    # nn.Linear(16, class_num))
    def init_hidden(self, batch_size):
        hidden_state = torch.zeros(self.n_layers,batch_size,self.n_hidden)
        cell_state = torch.zeros(self.n_layers,batch_size,self.n_hidden)
        self.hidden = (hidden_state, cell_state)

    def forward(self, x):
        bs = x.size(0)
        self.hidden = self.init_hidden(bs)
        x = x.squeeze(1)
        x = self.cnn_network(x)
        x = x.transpose(1, 2)
        lstm_out = self.l_lstm(x ,self.hidden)
        out, h = lstm_out
        # out = out[:, -1, :]
        out = torch.mean(out, 1)
        out = out.view(bs, -1)
        out = self.ann_network(out)
        return out

    def print_summary(self):
        print('Model Summary')
        summary(self, input_size=(1, self.input_size, self.time_len))
        print('\n')


class LSTM_Attn_Regression(nn.Module):
    def __init__(self, class_num=1, fs=None, ns=300):
        super(LSTM_Attn_Regression, self).__init__()
        input_size=26
        time_len=700 
        self.n_hidden = 50
        self.n_layers = 1
        self.time_len = ns
        self.input_size = input_size
        self.l_lstm = nn.LSTM(input_size = self.input_size, 
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers,
                                 batch_first=True)
        self.avg_pool = nn.AvgPool1d(700)
        self.ann_network = nn.Sequential(
                                    nn.Linear(self.n_hidden, class_num))
                                    # nn.ReLU(),
                                    # nn.Dropout(0.5),
                                    # nn.Linear(32, class_num))
    def init_hidden(self, batch_size):
        hidden_state = torch.zeros(self.n_layers,batch_size,self.n_hidden)
        cell_state = torch.zeros(self.n_layers,batch_size,self.n_hidden)
        self.hidden = (hidden_state, cell_state)

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state


    def forward(self, x):
        bs = x.size(0)
        self.hidden = self.init_hidden(bs)
        x = x.squeeze(1)
        x = x.view(bs, self.time_len, self.input_size) 
        out, (h, c) = self.l_lstm(x, self.hidden)
        attn_out = self.attention_net(out, h)

        # out = out[:, -1, :]
        # out = torch.mean(out, 1)

        out = attn_out.view(bs, -1)
        # out = out.view(bs, -1)
        out = self.ann_network(out)
        return out

    def print_summary(self):
        print('Model Summary')
        summary(self, input_size=(1, self.input_size, self.time_len))
        print('\n')


class CNN_LSTM_Attn_Regression(nn.Module):
    def __init__(self, class_num=1, fs=None, ns=None):
        super(CNN_LSTM_Attn_Regression, self).__init__()
        input_size=26
        
        self.n_layers = 1
        self.time_len = ns
        self.input_size = 50
        self.n_hidden = 25

        self.cnn_network = self.cnn_network = nn.Sequential(nn.Conv1d(in_channels=26, out_channels=int(self.input_size/2), kernel_size=5),
                            nn.ReLU(),
                            nn.MaxPool1d(4, 2),
                            nn.BatchNorm1d(int(self.input_size/2)))
        self.l_lstm = nn.LSTM(input_size = int(self.input_size/2), 
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers, 
                                 batch_first = True)
        self.avg_pool = nn.AvgPool1d(700)
        self.ann_network = nn.Sequential(
                                    nn.Linear(self.n_hidden, class_num))

    def init_hidden(self, batch_size):
        hidden_state = torch.zeros(self.n_layers,batch_size,self.n_hidden)
        cell_state = torch.zeros(self.n_layers,batch_size,self.n_hidden)
        self.hidden = (hidden_state, cell_state)

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state, soft_attn_weights

    def forward(self, x):
        bs = x.size(0)
        self.hidden = self.init_hidden(bs)
        x = x.squeeze(1)
        x = self.cnn_network(x)
        x = x.transpose(1, 2)
        lstm_out = self.l_lstm(x ,self.hidden)
        out, (h, c) = self.l_lstm(x, self.hidden)
        attn_out, weights = self.attention_net(out, h)
        out = attn_out.view(bs, -1)
        out = out.view(bs, -1)
        out = self.ann_network(out)
        return out

    def print_summary(self):
        print('Model Summary')
        summary(self, input_size=(1, self.input_size, self.time_len))
        print('\n')




if __name__ == "__main__":
    model = CNN_Regression_2D_700()
    print(model)
    x = torch.randn(5, 1, 26, 700)
    y_hat = model(x)
    print(y_hat.shape)

