import torch
import torch.nn as nn
import torch.nn.functional as F


# CharMLP模型类
class CharMLP(nn.Module):
    def __init__(self, args):
        super(CharMLP, self).__init__()
        
        # 使用注意力池化来处理可变长度序列
        # 注意力池化层 - 学习每个位置的重要性权重
        self.attention = nn.Sequential(
            nn.Linear(args.embedding_dim, args.embedding_dim // 2),
            nn.Tanh(),
            nn.Linear(args.embedding_dim // 2, 1),
            nn.Softmax(dim=1)
        )
        
        self.input_dim = args.embedding_dim
        
        # 多层感知机架构
        self.fc1 = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=args.dropout)
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=args.dropout)
        )
        
        self.fc3 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=args.dropout)
        )

        # 对于回归任务，最后一个线性层输出一个值
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        # 输入形状: (batch_size, embedding_dim, seq_length) - 与CNN相同的格式
        # 需要转换为: (batch_size, seq_length, embedding_dim) 用于注意力机制
        x = x.transpose(1, 2)  # (batch_size, seq_length, embedding_dim)
        
        # 注意力池化
        # 1. 计算注意力权重: (batch_size, seq_length, embedding_dim) -> (batch_size, seq_length, 1)
        attention_weights = self.attention(x)  # (batch_size, seq_length, 1)
        
        # 2. 加权求和: 每个位置按重要性加权
        # (batch_size, seq_length, embedding_dim) * (batch_size, seq_length, 1) -> (batch_size, seq_length, embedding_dim)
        weighted_x = x * attention_weights
        
        # 3. 求和得到固定长度表示: (batch_size, seq_length, embedding_dim) -> (batch_size, embedding_dim)
        x = torch.sum(weighted_x, dim=1)
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x) # 不使用softmax，直接输出回归值
        return x
