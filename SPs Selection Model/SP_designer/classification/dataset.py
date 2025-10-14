import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


class AJSDataset(torch.utils.data.Dataset):
    def __init__(self, data_frame, max_length=585, embedding_dim=16):
        self.data = data_frame
        # 替换数据中的\xa0字符为空字符串
        self.data['seq'] = self.data['seq'].apply(lambda x: x.replace('\xa0', ''))

        # 处理目标标签：转换成数值
        self.label_map = {'高': 1, '低': 0}
        self.data['label'] = self.data['label'].map(self.label_map)   # 将数据框 self.data 中 'label' 列的值根据 self.label_map 进行映射，将文字标签转换为数值形式。

        # 设置最大序列长度
        self.max_length = max_length

        # 获取唯一字符，建立字符到索引的映射
        unique_chars = sorted(set(''.join(self.data['seq'])))
        # unique_chars = set(''.join(self.data['seq']))
        self.char_to_idx = {char: idx+1 for idx, char in enumerate(unique_chars)}

        # 嵌入维度
        self.embedding_dim = embedding_dim

        # 嵌入层
        self.char_embedding = nn.Embedding(len(unique_chars)+1, embedding_dim)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取序列和标签
        seq = self.data.loc[idx, 'seq']
        label = self.data.loc[idx, 'label']

        # 将序列转换为索引，并填充至最大长度
        seq_indices = [self.char_to_idx[char] for char in seq]
        seq_indices = seq_indices[:self.max_length]  # 截断到最大长度
        seq_indices += [0] * (self.max_length - len(seq_indices))  # 填充

        # 嵌入字符
        seq_embedding = self.char_embedding(torch.tensor(seq_indices))

        # 创建填充掩码
        mask = (torch.tensor(seq_indices) != 0).unsqueeze(1)
        mask = mask.expand_as(seq_embedding)

        # 添加余弦位置嵌入，仅对非填充部分
        # seq_embedding = self.add_positional_encoding(seq_embedding, mask)

        return seq_embedding.T, label

    def add_positional_encoding(self, embeddings, mask):
        length, dim = embeddings.shape

        # 创建位置嵌入
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / dim))

        # 余弦和正弦混合
        pos_enc = torch.zeros_like(embeddings)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)

        # 仅对非填充部分添加位置嵌入
        # embeddings[mask] += pos_enc[mask]
        embeddings[mask.expand_as(embeddings)] += pos_enc[mask.expand_as(embeddings)]

        return embeddings


