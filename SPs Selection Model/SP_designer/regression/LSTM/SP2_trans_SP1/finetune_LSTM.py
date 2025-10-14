from torch.utils.data import DataLoader, SubsetRandomSampler, WeightedRandomSampler
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, root_mean_squared_error
from scipy.stats import pearsonr,spearmanr

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

# edits for google bolab compatibility and local environment

import sys
# Environment detection
try:
    import google.colab # type: ignore
    IN_COLAB = True
    print("🔹 Running in Google Colab")
    # Mount Google Drive if not already mounted
    try:
        from google.colab import drive # type: ignore
        drive.mount('/content/drive')
    except:
        pass
except:
    IN_COLAB = False
    print("🔹 Running in local environment")

# Set working directory based on environment
if IN_COLAB:
    # Adjust this path based on your Google Drive structure
    os.chdir('/content/drive/MyDrive')
    sys.path.append('/content/drive/MyDrive/Colab Notebooks/repo/SP_designer')
else:
    # For local environment, ensure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    # Add the project root to Python path
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
        
# Dynamic Matplotlib backend configuration
if IN_COLAB: # type: ignore
    matplotlib.use('Agg')
    print("Matplotlib backend set to 'Agg' for Colab")
else:
    matplotlib.use('TkAgg')
    print("Matplotlib backend set to 'TkAgg' for local environment")
    
from regression.LSTM.model_LSTM import CharLSTM
from regression.dataset import AJSDataset
# matplotlib.use('TkAgg')

# end of edits for google bolab compatibility and local environment

parser = argparse.ArgumentParser(description='LSTM training')
# data
parser.add_argument('--pre_train_path', metavar='DIR',
                    help='path to pre-training data csv', default=r'D:/SP_designer/data/SP2.xlsx')
parser.add_argument('--tune_path', metavar='DIR',
                    help='path to tune data csv', default=r'D:/SP_designer/data/SP1.xlsx')
parser.add_argument('--pre_model_path', metavar='DIR',
                    help='', default="D:/SP_designer/regression/LSTM/SP2_trans_SP1/CharLSTM_best.pth.tar")
parser.add_argument('--fine_model_path', metavar='DIR',
                    help='', default="D:/SP_designer/regression/LSTM/SP2_trans_SP1/fine_CharLSTM_best.pth.tar")
parser.add_argument('--max_length', type=int, default=50, help='')
parser.add_argument('--embedding_dim', type=int, default=30, help='')
parser.add_argument('--saved_values_path', metavar='DIR',
                    help='saved_values', default="D:/SP_designer/regression/LSTM/SP2_trans_SP1/saved_values.xlsx")

parser.add_argument('--unfreeze_after_epochs', type=int, default=50, help='冻结时间')
parser.add_argument('--tuning_strategy', type=int, default=2, help='1 冻结所有参数，只放开最后分类层'
                                                                   '2 全部参数都参与微调'
                                                                   '3 先冻结其他参数，一段时间后全部打开')

# learning
learn = parser.add_argument_group('Learning options')
learn.add_argument('--lr', type=float, default=0.03, help='initial learning rate [default: 0.0001]')
learn.add_argument('--epochs', type=int, default=300, help='number of epochs for train [default: 200]')
learn.add_argument('--batch_size', type=int, default=135, help='batch size for training [default: 64]')
learn.add_argument('--max_norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
learn.add_argument('--optimizer', default='SGD', help='Type of optimizer. SGD|Adam are supported [default: Adam]')
parser.add_argument("--lr_min", type=float, default=0.001, help="minimum learning rate for CosineAnnealingLR")
parser.add_argument("--T_max", type=int, default=100, help="maximum number of iterations for CosineAnnealingLR")

# model (text classifier)
lstm = parser.add_argument_group('Model options')
lstm.add_argument('--hidden_dim', type=int, default=6, help='')
lstm.add_argument('--num_layers', type=int, default=1, help='')



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
    checkpoint = torch.load(args.pre_model_path, map_location=args.device)
    model_state_dict = checkpoint['state_dict']

    model = CharLSTM(args)
    model.load_state_dict(model_state_dict)
    model.to(args.device)

    if args.tuning_strategy == 1:
        # 冻结所有参数，只放开最后分类层
        for param in model.parameters():
            param.requires_grad = False
        model.fc2.weight.requires_grad = True
        model.fc2.bias.requires_grad = True

    elif args.tuning_strategy == 2:
        # 全部参数都参与微调
        for param in model.parameters():
            param.requires_grad = True

    elif args.tuning_strategy == 3:
        # 先冻结其他参数，过100 epoch后全部打开
        for param in model.parameters():
            param.requires_grad = False
        model.fc2.weight.requires_grad = True
        model.fc2.bias.requires_grad = True
        model.unfreeze_after_epochs = args.unfreeze_after_epochs

    return model


def train_and_evaluate(model, train_loader, valid_loader, optimizer, scheduler, args):
    # 训练模型
    train(train_loader, valid_loader, model, optimizer, scheduler, args)

    model.eval()
    checkpoint = torch.load(args.fine_model_path, map_location=args.device)
    model_state_dict = checkpoint['state_dict']

    model_best = CharLSTM(args)
    model_best.load_state_dict(model_state_dict)
    model_best.to(args.device)
    # 验证模型
    predictions, true_labels = [], []
    with torch.no_grad():
        for data in valid_loader:
            inputs, labels = data
            outputs = model_best(inputs.to(args.device))

            predictions.extend(outputs.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    print("————————")
    print("Predictions:", predictions)
    print("True Labels:", true_labels)
    print("————————")
    pearson_corr, spearman_corr, r_squared, rmse = evaluate_regression(true_labels, predictions)
    print(f"Pearson Correlation: {pearson_corr}")
    print(f'Spearman Correlation: {spearman_corr}')
    print(f"R^2 Score: {r_squared}")
    print(f"RMSE: {rmse}")

    return pearson_corr, spearman_corr, r_squared, rmse, predictions, true_labels


def train(train_dataloader, valid_dataloader, model, optimizer, scheduler, args):
    start_epoch = 1
    start_iter = 1
    best_loss = None

    model.train()

    for epoch in range(start_epoch, args.epochs + 1):
        if hasattr(model, 'unfreeze_after_epochs') and epoch == model.unfreeze_after_epochs:
            print("—————— 参数已解冻 ——————")
            # 解冻所有参数
            for param in model.parameters():
                param.requires_grad = True
        total_loss = 0.0
        for i_batch, data in enumerate(train_dataloader, start=start_iter):
            inputs, target = data
            inputs, target = inputs.to(args.device), target.to(args.device)

            logit = model(inputs)
            loss = torch.nn.MSELoss()(logit, target.view(-1, 1))

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            i_batch += 1
        # validation
        val_loss = eval(valid_dataloader, model, epoch, i_batch, optimizer, args)
        if best_loss is None or val_loss < best_loss:
            file_path = args.fine_model_path
            # print("\r=> found better validated model, saving to %s" % file_path)
            save_checkpoint(model,
                            {'epoch': epoch,
                             'optimizer': optimizer.state_dict(),
                             'best_loss': best_loss},
                            file_path)
            best_loss = val_loss
        # print('\n')
    print("————————————————", best_loss)




def eval(valid_dataloader, model, epoch_train, batch_train, optimizer, args):
    model.eval()
    total_loss = 0.0

    for i_batch, (data) in enumerate(valid_dataloader):
        inputs, target = data
        inputs, target = inputs.to(args.device), target.to(args.device)
        logit = model(inputs)
        loss = torch.nn.MSELoss()(logit, target.view(-1, 1))
        total_loss += loss.item()

    avg_loss = total_loss / len(valid_dataloader)
    model.train()
    # print('\nEvaluation - loss: {:.6f}  lr: {:.5f}) '.format(avg_loss,optimizer.state_dict()['param_groups'][0]['lr']))



    return avg_loss

def save_checkpoint(model, state, filename):
    state['state_dict'] = model.state_dict()
    torch.save(state, filename)

def evaluate_regression(true_labels, predictions):
    predictions = [float(pred[0]) for pred in predictions]
    pearson_corr, _ = pearsonr(predictions, true_labels)
    spearman_corr, _=spearmanr(predictions,true_labels)
    r_squared = r2_score(true_labels, predictions)
    rmse = root_mean_squared_error(true_labels, predictions)
    return pearson_corr, spearman_corr, r_squared, rmse

def main():
    # 解析命令行参数
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    # 数据加载
    original_data = pd.read_excel(args.tune_path)
    dataset = AJSDataset(original_data, args.max_length, args.embedding_dim)

    # 设置 k 折交叉验证
    k = 10
    kf = KFold(n_splits=k, shuffle=True)

    # 准备存储回归评价指标的变量

    pearson_corrs = []
    spearman_corrs = []
    r_squareds = []
    rmses = []
    all_true_values=[]
    all_pred_values=[]

    # K 折循环
    for fold, (train_index, valid_index) in enumerate(kf.split(dataset)):
        all_predictions = []
        all_true_labels = []
        train_sampler = SubsetRandomSampler(train_index)
        valid_sampler = SubsetRandomSampler(valid_index)

        Train_dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler,
                                      num_workers=0)
        Valid_dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=valid_sampler,
                                      num_workers=0)
        # 打印数据加载器中的数据数量
        print(f"{fold + 1} 训练数据量:", len(Train_dataloader))
        print(f"{fold + 1} 验证数据量:", len(Valid_dataloader))

        # 模型初始化
        # 加载预训练模型
        model = load_model(args)
        optimizer = optim.Adam(model.parameters(), lr=args.lr) if args.optimizer == 'Adam' else optim.SGD(
            model.parameters(), lr=args.lr, momentum=0.9)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.lr_min)

        # 调用训练和验证
        pearson_corr, spearman_corr, r_squared, rmse, predictions, true_labels = train_and_evaluate(model, Train_dataloader,
                                                                                       Valid_dataloader, optimizer,
                                                                                       scheduler, args)

        # 存储结果
        all_true_labels.extend(true_labels)
        all_predictions.extend(predictions)

        # 计算评价指标
        pearson_corr, spearman_corr, r_squared, rmse = evaluate_regression(np.array(all_true_labels), np.array(all_predictions))
        pearson_corrs.append(pearson_corr)
        spearman_corrs.append(spearman_corr)
        r_squareds.append(r_squared)
        rmses.append(rmse)

        print(f'Fold {fold + 1}: Pearson Correlation: {pearson_corr}, R^2 Score: {r_squared}, RMSE: {rmse}')
        print()

        all_true_values.extend(all_true_labels)
        all_pred_values.extend([value[0] for value in all_predictions])

    # 计算交叉验证的平均评价指标
    avg_pearson = np.mean(pearson_corrs)
    avg_spearman = np.mean(spearman_corrs)
    avg_r_squared = np.mean(r_squareds)
    avg_rmse = np.mean(rmses)

    print(f"平均Pearson Correlation: {avg_pearson:.2f}")
    print(f'平均Spearman Correlation: {avg_spearman:.2f}')
    print(f"平均R^2 Score: {avg_r_squared:.2f}")
    print(f"平均RMSE: {avg_rmse:.2f}")

    data = pd.DataFrame({'真实值': all_true_values, '预测值': all_pred_values})
    data.to_excel(args.saved_values_path, index=False)

if __name__ == '__main__':
    set_seed(42)
    main()
