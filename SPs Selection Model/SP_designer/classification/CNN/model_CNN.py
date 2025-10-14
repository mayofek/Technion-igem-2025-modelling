import torch
import torch.nn as nn
import torch.nn.functional as F


# CharCNN模型类
class CharCNN(nn.Module):
    def __init__(self, args):
        super(CharCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(args.embedding_dim, 8, kernel_size=3, stride=1),
            nn.AvgPool1d(kernel_size=3, stride=1),
            nn.ReLU(),  # 激活函数
            nn.Dropout(p=args.dropout)
        )

        # self.conv2 = nn.Sequential(
        #     nn.Conv1d(8, 4, kernel_size=3, stride=1),
        #     nn.AvgPool1d(kernel_size=3, stride=1),
        #     nn.ReLU() ,
        #     nn.Dropout(p=args.dropout)
        # )
        # self.conv3 = nn.Sequential(
        #     nn.Conv1d(4, 2, kernel_size=3, stride=1),
        #     nn.AvgPool1d(kernel_size=3, stride=1),
        #     nn.ReLU(),
        #     # nn.Dropout(p=args.dropout)
        # )


        # 全连接层
        self.fc1 = nn.Sequential(
            nn.Linear(368, 128),
            nn.ReLU(),
            nn.Dropout(p=args.dropout)
        )

        # self.fc2 = nn.Sequential(
        #     nn.Linear(512, 16),
        #     nn.ReLU(),
        #     nn.Dropout(p=args.dropout)
        # )

        self.fc2 = nn.Linear(128, args.num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
