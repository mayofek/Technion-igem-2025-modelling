from torch.utils.data import DataLoader, SubsetRandomSampler, WeightedRandomSampler
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve

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
    
from classification.LSTM.model_LSTM import CharLSTM
from classification.dataset import AJSDataset
# matplotlib.use('TkAgg')

# end of edits for google bolab compatibility and local environment

parser = argparse.ArgumentParser(description='LSTM classifier training')
# data
parser.add_argument('--train_path', metavar='DIR',
                    help='path to training data', default=r'D:/SP_designer/data/SP2.xlsx')

parser.add_argument('--test_path', metavar='DIR',
                    help='path to testing data', default=r'D:/SP_designer/data/SP1.xlsx')

parser.add_argument('--model_path', metavar='DIR',
                    help='', default="D:/SP_designer/classification/LSTM/only_SP2/CharLSTM_best.pth.tar")


parser.add_argument('--max_length', type=int, default=50, help='')
parser.add_argument('--embedding_dim', type=int, default=8, help='')
parser.add_argument('--saved_values_path', metavar='DIR',
                    help='saved_values', default="D:/SP_designer/classification/LSTM/only_SP2/saved_values.xlsx")

# learning
learn = parser.add_argument_group('Learning options')
learn.add_argument('--lr', type=float, default=0.1, help='initial learning rate [default: 0.0001]')
learn.add_argument('--epochs', type=int, default=300, help='number of epochs for train [default: 200]')
learn.add_argument('--batch_size', type=int, default=121, help='batch size for training [default: 64]')
learn.add_argument('--max_norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
learn.add_argument('--optimizer', default='SGD', help='Type of optimizer. SGD|Adam are supported [default: Adam]')
parser.add_argument("--lr_min", type=float, default=0.001, help="minimum learning rate for CosineAnnealingLR")
parser.add_argument("--T_max", type=int, default=20, help="maximum number of iterations for CosineAnnealingLR")

# model (text classifier)
lstm = parser.add_argument_group('Model options')
lstm.add_argument('--num_classes', type=int, default=2, help='')
lstm.add_argument('--hidden_dim', type=int, default=12, help='')
lstm.add_argument('--num_layers', type=int, default=1, help='')

# experiment options
experiment = parser.add_argument_group('Experiment options')
experiment.add_argument('--checkpoint', dest='checkpoint', default=True, action='store_true',
                        help='Enables checkpoint saving of model')
experiment.add_argument('--save_folder', default='D:/SP_designer/classification/LSTM/only_SP2',
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
    pred_targets_prob = []
    with torch.no_grad():
        for data in valid_loader:
            inputs, labels = data
            inputs = inputs.transpose(1, 2)
            outputs = best_model(inputs.to(args.device))
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            pred_targets_prob.extend(torch.softmax(outputs, dim=1).cpu().tolist())
    # 计算准确率、精确率、召回率、F1、混淆矩阵
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    f1 = f1_score(true_labels, predictions, average='weighted')

    return accuracy, precision, recall, f1, predictions, true_labels, pred_targets_prob


def train(train_dataloader, valid_dataloader, model, optimizer, scheduler, args):
    start_epoch = 1
    start_iter = 1
    best_acc = None

    model.train()

    for epoch in range(start_epoch, args.epochs + 1):
        avg_corrects, avg_loss = 0.0, 0.0
        for i_batch, data in enumerate(train_dataloader, start=start_iter):
            inputs, target = data
            inputs = inputs.transpose(1, 2)
            inputs, target = inputs.to(args.device), target.to(args.device)

            logit = model(inputs)
            # logit = F.softmax(logit)

            loss = F.cross_entropy(logit, target)
            avg_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            scheduler.step()

            corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
            accuracy = 100.0 * corrects / args.batch_size
            avg_corrects += corrects

            i_batch += 1
        # print('Epoch[{}]] - loss: {:.6f}  lr: {:.5f}  acc: {:.3f}% '.format(epoch, avg_loss / len(train_dataloader),
        #                                                                     optimizer.state_dict()['param_groups'][0]['lr'],
        #                                                                     avg_corrects/len(train_dataloader)))
        # validation
        val_loss, val_acc = eval(valid_dataloader, model, epoch, i_batch, optimizer, args)
        # save best validation epoch model
        if best_acc is None or val_acc > best_acc:
            file_path = '%s/CharLSTM_best.pth.tar' % (args.save_folder)
            # print("\r=> found better validated model, saving to %s" % file_path)
            save_checkpoint(model,
                            {'epoch': epoch,
                             'optimizer': optimizer.state_dict(),
                             'best_acc': best_acc},
                            file_path)
            best_acc = val_acc
        # print('\n')
    print("————————————————", best_acc)


def eval(valid_dataloader, model, epoch_train, batch_train, optimizer, args):
    model.eval()
    corrects, avg_loss, accumulated_loss, size = 0, 0, 0, 0
    predicates_all, target_all = [], []
    for i_batch, (data) in enumerate(valid_dataloader):
        inputs, target = data
        size += len(target)
        inputs = inputs.transpose(1, 2)
        inputs, target = inputs.to(args.device), target.to(args.device)
        logit = model(inputs)
        predicates = torch.max(logit, 1)[1].view(target.size()).data
        accumulated_loss += F.cross_entropy(logit, target, reduction='sum').data
        corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
        predicates_all += predicates.cpu().numpy().tolist()
        target_all += target.data.cpu().numpy().tolist()

    avg_loss = accumulated_loss / size
    accuracy = 100.0 * corrects / size
    model.train()
    # print('\nEvaluation - loss: {:.6f}  lr: {:.5f}  acc: {:.3f}% ({}/{}) '.format(avg_loss,
    #                                                                               optimizer.state_dict()[
    #                                                                                   'param_groups'][0]['lr'],
    #                                                                               accuracy,
    #                                                                               corrects,
    #                                                                               size))
    # print_f_score(predicates_all, target_all)
    if args.log_result:
        with open(os.path.join(args.save_folder, 'result.csv'), 'a') as r:
            r.write('\n{:d},{:d},{:.5f},{:.2f},{:f}'.format(epoch_train,
                                                            batch_train,
                                                            avg_loss,
                                                            accuracy,
                                                            optimizer.state_dict()['param_groups'][0]['lr']))

    return avg_loss, accuracy


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
            r.write('{:s},{:s},{:s},{:s},{:s}'.format('epoch', 'batch', 'loss', 'acc', 'lr'))


    # 数据加载
    train_data = pd.read_excel(args.train_path)
    test_data = pd.read_excel(args.test_path)
    train_set = AJSDataset(train_data, args.max_length, args.embedding_dim)
    test_set = AJSDataset(test_data, args.max_length, args.embedding_dim)

    class_counts_train = {}
    for idx in range(train_set.__len__()):
        _, label = train_set[idx]
        if label in class_counts_train:
            class_counts_train[label] += 1
        else:
            class_counts_train[label] = 1
    print("训练集类别分布:", class_counts_train)
    weights = []
    for idx in range(train_set.__len__()):
        _, label = train_set[idx]
        class_weight = 1.0 / class_counts_train[label]
        weights.append(class_weight)

    train_sampler = WeightedRandomSampler(weights, train_set.__len__())


    # 准备存储结果的变量
    all_true_labels = []
    all_predictions = []
    all_pred_prob = []
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []


    # 设置 k 折交叉验证
    k = 10
    kf = KFold(n_splits=k, shuffle=True)
    # K 折循环
    for fold, (_, valid_index) in enumerate(kf.split(test_set)):
        valid_sampler = SubsetRandomSampler(valid_index)

        Train_dataloader = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler,
                                      num_workers=0)
        Valid_dataloader = DataLoader(test_set, batch_size=args.batch_size, sampler=valid_sampler,
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
        accuracy, precision, recall, f1, predictions, true_labels, pred_targets_prob= train_and_evaluate(model, Train_dataloader, Valid_dataloader, optimizer, scheduler, args)

        # 存储结果
        all_true_labels.extend(true_labels)
        all_predictions.extend(predictions)
        all_pred_prob.extend(pred_targets_prob)

        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)


        print(f'Fold {fold + 1}: Done')

        print(f"{fold + 1} 准确率: {accuracy:.2f}")
        print(f"{fold + 1} 精确率: {precision:.2f}")
        print(f"{fold + 1} 召回率: {recall:.2f}")
        print(f"{fold + 1} F1: {f1:.2f}")
        print()


    # 计算交叉验证的平均分类指标
    avg_accuracy = np.mean(accuracy_list)
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_f1 = np.mean(f1_list)
    cm = confusion_matrix(all_true_labels, all_predictions)
    print(cm)

    print(f"平均准确率: {avg_accuracy:.2f}")
    print(f"平均精确率: {avg_precision:.2f}")
    print(f"平均召回率: {avg_recall:.2f}")
    print(f"平均F1: {avg_f1:.2f}")

    prob_0=[prob[0] for prob in all_pred_prob]
    prob_1=[prob[1] for prob in all_pred_prob]
    data = pd.DataFrame({'真实标签值': all_true_labels, '预测标签值': all_predictions, '概率值0': prob_0, '概率值1': prob_1})
    data.to_excel(args.saved_values_path, index=False)


if __name__ == '__main__':
    set_seed(42)
    main()

