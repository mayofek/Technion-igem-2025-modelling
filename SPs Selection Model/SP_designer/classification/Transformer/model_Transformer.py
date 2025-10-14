import torch
import torch.nn as nn
import torch.nn.functional as F



class CharTransformer(nn.Module):
    def __init__(self, args):
        super(CharTransformer, self).__init__()
        self.embedding_dim = args.embedding_dim
        self.hidden_dim = args.hidden_dim
        self.num_encoder_layers = args.num_encoder_layers
        self.num_heads = args.num_heads

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim,
            batch_first=True)

        self.transformer = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=self.num_encoder_layers)

        self.fc1 = nn.Linear(self.embedding_dim, args.num_classes)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        # print(x.shape)
        transformer_out = self.transformer(x)
        transformer_out = transformer_out[:, -1, :]  # 取最后一个时刻的输出作为全连接层的输入
        out = self.fc1(transformer_out)
        out = self.softmax(out)
        return out
