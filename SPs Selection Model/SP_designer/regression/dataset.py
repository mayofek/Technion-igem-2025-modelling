import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class AJSDataset(Dataset):
    def __init__(self, data_frame, max_length=585, embedding_dim=16):
        self.data = data_frame
        # 替换数据中的\xa0字符为空字符串
        self.data['seq'] = self.data['seq'].apply(lambda x: x.replace('\xa0', ''))


        # 设置最大序列长度
        self.max_length = max_length
        # 获取唯一字符，建立字符到索引的映射
        unique_chars = sorted(set(''.join(self.data['seq'])))
        # unique_chars = set(''.join(self.data['seq']))
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(unique_chars)}
        # 嵌入维度
        self.embedding_dim = embedding_dim
        # 嵌入层
        self.char_embedding = nn.Embedding(len(unique_chars) + 1, embedding_dim)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取序列和标签
        seq = self.data.loc[idx, 'seq']
        label = self.data.loc[idx, 'target']

        # 将序列转换为索引，并填充至最大长度
        seq_indices = [self.char_to_idx[char] for char in seq]
        seq_indices = seq_indices[:self.max_length]  # 截断到最大长度
        seq_indices += [0] * (self.max_length - len(seq_indices))  # 填充

        # 嵌入字符
        seq_embedding = self.char_embedding(torch.tensor(seq_indices))

        return seq_embedding.T, torch.tensor(label, dtype=torch.float)


