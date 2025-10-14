import torch
import torch.nn as nn
import torch.nn.functional as F


# CharCNN模型类
class CharCNN(nn.Module):
    def __init__(self, args):
        super(CharCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(args.embedding_dim, 3, kernel_size=3, stride=1),
            # nn.AvgPool1d(kernel_size=3, stride=1),
            nn.ReLU(),  # 激活函数
            # nn.Dropout(p=args.dropout)
        )

        # 全连接层
        self.fc1 = nn.Sequential(
            nn.Linear(144, 128),
            nn.ReLU(),
            # nn.Dropout(p=args.dropout)
        )

        # 对于回归任务，最后一个线性层输出一个值
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x) # 不使用softmax，直接输出回归值
        return x
