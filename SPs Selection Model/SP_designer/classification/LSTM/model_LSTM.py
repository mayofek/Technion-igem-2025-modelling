import torch
import torch.nn as nn
import torch.nn.functional as F



class CharLSTM(nn.Module):
    def __init__(self, args):
        super(CharLSTM, self).__init__()
        self.embedding_dim = args.embedding_dim
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_dim, args.num_classes)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        # print(x.shape)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # 取最后一个时刻的输出作为全连接层的输入
        out = self.fc1(lstm_out)
        out = self.softmax(out)
        return out
