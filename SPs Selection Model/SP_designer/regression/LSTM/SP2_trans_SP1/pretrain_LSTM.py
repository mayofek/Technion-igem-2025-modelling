from torch.utils.data import DataLoader, SubsetRandomSampler, WeightedRandomSampler
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, root_mean_squared_error
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
import matplotlib
from torch import optim
import numpy as np
import pandas as pd
import argparse
import random
import errno
import torch
import os

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
parser.add_argument('--model_path', metavar='DIR',
                    help='', default="D:/SP_designer/regression/LSTM/SP2_trans_SP1/CharLSTM_best.pth.tar")
parser.add_argument('--max_length', type=int, default=50, help='')
parser.add_argument('--embedding_dim', type=int, default=30, help='')

# learning
learn = parser.add_argument_group('Learning options')
learn.add_argument('--lr', type=float, default=0.03, help='initial learning rate [default: 0.0001]')
learn.add_argument('--epochs', type=int, default=200, help='number of epochs for train [default: 200]')
learn.add_argument('--batch_size', type=int, default=70, help='batch size for training [default: 64]')
learn.add_argument('--max_norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
learn.add_argument('--optimizer', default='SGD', help='Type of optimizer. SGD|Adam are supported [default: Adam]')
parser.add_argument("--lr_min", type=float, default=0.001, help="minimum learning rate for CosineAnnealingLR")
parser.add_argument("--T_max", type=int, default=50, help="maximum number of iterations for CosineAnnealingLR")

# model (text classifier)
lstm = parser.add_argument_group('Model options')
lstm.add_argument('--hidden_dim', type=int, default=6, help='')
lstm.add_argument('--num_layers', type=int, default=1, help='')

# experiment options
experiment = parser.add_argument_group('Experiment options')
experiment.add_argument('--checkpoint', dest='checkpoint', default=True, action='store_true',
                        help='Enables checkpoint saving of model')
experiment.add_argument('--save_folder', default='D:/SP_designer/regression/LSTM/SP2_trans_SP1',
                        help='Location to save epoch models, training configurations and results.')
experiment.add_argument('--log_result', default=True, action='store_true', help='Store experiment result')

def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # Numpy module
    torch.manual_seed(seed)  # PyTorch
    torch.cuda.manual_seed(seed)  # For CUDA
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    torch.backends.cudnn.deterministic = True  # Makes computations deterministic
    torch.backends.cudnn.benchmark = True  # If True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.


def train_and_evaluate(model, train_loader, valid_loader, optimizer, scheduler, args):
    # 训练模型
    train(train_loader, valid_loader, model, optimizer, scheduler, args)

    model.eval()
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model_state_dict = checkpoint['state_dict']

    best_model = CharLSTM(args)
    best_model.load_state_dict(model_state_dict)
    best_model.to(args.device)
    # 验证模型
    predictions, true_labels = [], []
    with torch.no_grad():
        for data in valid_loader:
            inputs, labels = data
            outputs = best_model(inputs.to(args.device))

            predictions.extend(outputs.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    print("————————")
    print("Predictions:", predictions)
    print("True Labels:", true_labels)
    print("————————")
    pearson_corr, r_squared, rmse = evaluate_regression(true_labels, predictions)
    print(f"Pearson Correlation: {pearson_corr}")
    print(f"R^2 Score: {r_squared}")
    print(f"RMSE: {rmse}")

    return pearson_corr, r_squared, rmse, predictions, true_labels


def train(train_dataloader, valid_dataloader, model, optimizer, scheduler, args):
    start_epoch = 1
    start_iter = 1
    best_loss = None

    model.train()

    for epoch in range(start_epoch, args.epochs + 1):
        total_loss = 0.0
        for i_batch, data in enumerate(train_dataloader, start=start_iter):
            inputs, target = data
            inputs, target = inputs.to(args.device), target.to(args.device)

            logit = model(inputs)
            # print("logit:", logit)
            loss = torch.nn.MSELoss()(logit, target.view(-1, 1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            i_batch += 1
        avg_loss = total_loss / len(train_dataloader)
        # print('Epoch[{}]] - loss: {:.6f}  lr: {:.5f}'.format(epoch, avg_loss,
        #                                                      optimizer.state_dict()['param_groups'][0]['lr']))
        # validation
        val_loss = eval(valid_dataloader, model, epoch, i_batch, optimizer, args)
        # save best validation epoch model
        if best_loss is None or val_loss < best_loss:
            file_path = '%s/CharLSTM_best.pth.tar' % (args.save_folder)
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
        # print("logit:", logit)
        loss = torch.nn.MSELoss()(logit, target.view(-1, 1))
        total_loss += loss.item()

    avg_loss = total_loss / len(valid_dataloader)
    model.train()
    # print('\nEvaluation - loss: {:.6f}  lr: {:.5f}) '.format(avg_loss,optimizer.state_dict()['param_groups'][0]['lr']))

    if args.log_result:
        with open(os.path.join(args.save_folder, 'result.csv'), 'a') as r:
            r.write('\n{:d},{:d},{:.5f},{:f}'.format(epoch_train,
                                                            batch_train,
                                                            avg_loss,
                                                            optimizer.state_dict()['param_groups'][0]['lr']))

    return avg_loss

def evaluate_regression(true_labels, predictions):
    predictions = [float(pred[0]) for pred in predictions]
    pearson_corr, _ = pearsonr(predictions, true_labels)
    r_squared = r2_score(true_labels, predictions)
    rmse = root_mean_squared_error(true_labels, predictions)
    return pearson_corr, r_squared, rmse

def save_checkpoint(model, state, filename):
    state['state_dict'] = model.state_dict()
    torch.save(state, filename)


def main():
    # 解析命令行参数
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # make save folder
    try:
        os.makedirs(args.save_folder)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Directory already exists.')
        else:
            raise
    # configuration
    print("\nConfiguration:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}:".format(attr.capitalize().replace('_', ' ')).ljust(25) + "{}".format(value))
    # log result
    if args.log_result:
        with open(os.path.join(args.save_folder, 'result.csv'), 'w') as r:
            r.write('{:s},{:s},{:s},{:s}'.format('epoch', 'batch', 'loss', 'lr'))


    # 数据加载
    original_data = pd.read_excel(args.pre_train_path)
    dataset = AJSDataset(original_data, args.max_length, args.embedding_dim)

    # 设置 k 折交叉验证
    k = 10
    kf = KFold(n_splits=k, shuffle=True)

    # 准备存储回归评价指标的变量
    pearson_corrs = []
    r_squareds = []
    rmses = []

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
        model = CharLSTM(args).to(args.device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr) if args.optimizer == 'Adam' else optim.SGD(
            model.parameters(), lr=args.lr, momentum=0.9)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.lr_min)

        # 调用训练和验证
        pearson_corr, r_squared, rmse, predictions, true_labels= train_and_evaluate(model, Train_dataloader, Valid_dataloader, optimizer, scheduler, args)

        # 存储结果
        all_true_labels.extend(true_labels)
        all_predictions.extend(predictions)

        # 计算评价指标
        pearson_corr, r_squared, rmse = evaluate_regression(np.array(all_true_labels), np.array(all_predictions))
        pearson_corrs.append(pearson_corr)
        r_squareds.append(r_squared)
        rmses.append(rmse)

        print(f'Fold {fold + 1}: Pearson Correlation: {pearson_corr}, R^2 Score: {r_squared}, RMSE: {rmse}')
        print()

    # 计算交叉验证的平均评价指标
    avg_pearson = np.mean(pearson_corrs)
    avg_r_squared = np.mean(r_squareds)
    avg_rmse = np.mean(rmses)

    print(f"平均Pearson Correlation: {avg_pearson:.2f}")
    print(f"平均R^2 Score: {avg_r_squared:.2f}")
    print(f"平均RMSE: {avg_rmse:.2f}")



if __name__ == '__main__':
    set_seed(42)
    main()