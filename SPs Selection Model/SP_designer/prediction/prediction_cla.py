from torch.utils.data import DataLoader, SubsetRandomSampler, WeightedRandomSampler
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve

import torch
from torch import nn, optim
import numpy as np
import pandas as pd
import argparse
import random
import errno
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


from classification.CNN.model_CNN import CharCNN

from classification.dataset import AJSDataset

# COMMENTED OUT: Original hardcoded paths
# source_path='D:/SP_designer/prediction/generated_sp.xlsx'
# temp_data=pd.read_excel(source_path)
# seq_list=temp_data['序列']
# label_list=['低' for i in range(len(seq_list))]
# temp_data = pd.DataFrame(
#     {'seq': seq_list, 'label': label_list})
# temp_data.to_excel('./temp_data.xlsx', index=False)

# NEW: These will be moved to main() to use command line arguments


parser = argparse.ArgumentParser(description='CNN classifier training')
# data
parser.add_argument('--test_path', metavar='DIR',
                    help='path to tune data csv', default=r'D:/SP_designer/prediction/temp_data.xlsx')
parser.add_argument('--model_path', metavar='DIR',
                    help='', default="D:/SP_designer/prediction/fine_CharCNN_best_cla.pth.tar")
parser.add_argument('--max_length', type=int, default=50, help='')
parser.add_argument('--embedding_dim', type=int, default=8, help='')
# NEW: Add arguments for all customizable paths
parser.add_argument('--input_path', metavar='DIR',
                    help='path to input Excel file', default=r'D:/SP_designer/prediction/generated_sp.xlsx')
parser.add_argument('--output_path', metavar='DIR',
                    help='path to output Excel file', default=r'D:/SP_designer/prediction/generated_sp.xlsx')
parser.add_argument('--temp_file', metavar='DIR',
                    help='path to temporary data file', default=r'./temp_data.xlsx')


# model (text classifier)
cnn = parser.add_argument_group('Model options')
cnn.add_argument('--num_classes', type=int, default=2, help='')
cnn.add_argument('--dropout', type=float, default=0.3, help='无所谓，不会调用')

def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # Numpy module
    torch.manual_seed(seed)  # PyTorch
    torch.cuda.manual_seed(seed)  # For CUDA
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    torch.backends.cudnn.deterministic = True  # Makes computations deterministic
    torch.backends.cudnn.benchmark = True  # If True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.

def load_model(args):
    # 加载文件并提取模型状态
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model_state_dict = checkpoint['state_dict']

    model = CharCNN(args)
    model.load_state_dict(model_state_dict)
    model.to(args.device)

    return model



def test(dataloader, model, args):
    res_list=[]
    pred_targets_prob=[]
    model.eval()
    for i_batch, (data) in enumerate(dataloader):
        inputs, target = data
        inputs = inputs.to(args.device)
        logit = model(inputs)
        _, preds = torch.max(logit, 1)
        preds = preds.cpu()
        output = "高" if preds == 1 else "低"
        print(f"output: {output}")
        res_list.append(output)
        pred_targets_prob.extend(torch.softmax(logit, dim=1).cpu().tolist())
    print(pred_targets_prob)
    return res_list


def main():
    # 解析命令行参数
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # NEW: Use command line arguments for all paths
    source_path = args.input_path
    output_path = args.output_path
    temp_data=pd.read_excel(source_path)
    seq_list=temp_data['序列']
    label_list=['低' for i in range(len(seq_list))]
    temp_data = pd.DataFrame(
        {'seq': seq_list, 'label': label_list})
    temp_data.to_excel(args.temp_file, index=False)

    # 数据加载
    original_data = pd.read_excel(args.test_path)
    dataset = AJSDataset(original_data, args.max_length, args.embedding_dim)

    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)


    # 模型初始化
    # 加载预训练模型
    model = load_model(args)

    res_list=test(dataloader, model, args)

    # COMMENTED OUT: Original hardcoded output
    # save_data=pd.read_excel(source_path)
    # save_data['预测类别']=res_list
    # save_data.to_excel(source_path, index=False)
    
    # NEW: Use separate output path
    save_data=pd.read_excel(source_path)
    save_data['预测类别']=res_list
    save_data.to_excel(output_path, index=False)

    # COMMENTED OUT: was removing test_path, now removing temp_file
    # os.remove(args.test_path)
    
    # NEW: Remove the temporary file we created
    os.remove(args.temp_file)


if __name__ == '__main__':
    set_seed(42)
    main()

