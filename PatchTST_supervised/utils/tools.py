import torch
import matplotlib.pyplot as plt
import time
import math
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from sklearn.utils import resample

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from utils.timefeatures import time_features
import warnings
from scipy.stats import norm, expon, lognorm, gamma, beta
import scipy.stats as stats

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate * 0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate * 0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate * 0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate * 0.1}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

# exp_main:
# def visual(true, preds=None, name='./pic/test.pdf'):
#     """
#     Results visualization
#     """
#     plt.figure()
#     plt.plot(true, label='GroundTruth', linewidth=2)
#     if preds is not None:
#         plt.plot(preds, label='Prediction', linewidth=2)
#     plt.legend()
#     plt.savefig(name, bbox_inches='tight')


# exp_main_1:
# def visual(y_true, upper_bound, lower_bound, y_pred, mont_upper, mont_lower, boot_upper, boot_lower=None, name='./pic/test.pdf'):
#     """
#     Results visualization
#     """
#     fig, ax = plt.subplots()
#     ax.plot(y_true,  color='blue', marker='')
#     # ax.plot(y_true, label='True Values', color='blue', marker='')
#     if y_pred is not None:
#         ax.fill_between(np.arange(len(y_pred)), lower_bound, upper_bound, alpha=0.2,
#                         color='gray')
#         # ax.fill_between(np.arange(len(y_pred)), lower_bound, upper_bound, alpha=0.2, label='Confidence Interval',
#         #                 color='gray')
#         # ax.fill_between(np.arange(len(y_pred)), mont_lower, mont_upper, alpha=0.2, label='mont Interval',
#         #                 color='blue')
#         # ax.fill_between(np.arange(len(y_pred)), boot_lower, boot_upper, alpha=0.2, label='boot Interval',
#         #                 color='orange')
#         # ax.plot(y_pred, label='Predicted Values', color='green', marker='')
#     ax.set_xlabel('Time')
#     ax.set_ylabel('Values')
#     ax.legend()
#     plt.savefig(name, bbox_inches='tight')

def visual(y_true, upper_bound, lower_bound, y_pred=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    fig, ax = plt.subplots()
    ax.plot(y_true, label='True Values', color='blue', marker='')
    if y_pred is not None:
        ax.fill_between(np.arange(len(y_pred)), lower_bound, upper_bound, alpha=0.2, label='Confidence Interval',
                        color='gray')
        ax.plot(y_pred, label='Predicted Values', color='green', marker='')
    ax.set_xlabel('Time')
    ax.set_ylabel('Values')
    ax.legend()
    plt.savefig(name, bbox_inches='tight')


def test_params_flop(model, x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
        print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    from ptflops import get_model_complexity_info
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
        # print('Flops:' + flops)
        # print('Params:' + params)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))


def Conference_interval(true, pred, conf, mse):
    conf_list = [0.95, 0.9, 0.85, 0.8]
    z = [1.96, 1.64, 1.44, 1.29]
    index = conf_list.index(conf)
    # 预测区间
    # avg = np.mean(true[0,:,-1])
    # variance = (true[0,:,-1] - avg)**2
    # stand = math.sqrt(np.sum(variance)/len(true[0,:,-1]))

    # 置信区间
    # stand = math.sqrt(np.sum(mse))
    # up_line = pred[0,:,-1] + stand*z[index]
    # low_line = pred[0,:,-1] - stand*z[index]
    # return up_line,low_line

    # 定义 MCMC 采样的次数
    num_samples = 1000
    lower_2down = []
    upper_2down = []
    samples = np.zeros((num_samples, len(pred[0, :, -1])))
    avg = np.mean(pred[0, :, -1])
    variance = (pred[0, :, -1] - avg) ** 2
    stand = math.sqrt(np.sum(variance) / len(pred[0, :, -1]))
    # larger_std_dev = 2 * np.sqrt(mse)

    for i in range(num_samples):
        random_values = np.random.normal(0, 0.1*np.std(pred[0,:,-1]), size=len(pred[0,:,-1]))
        # random_values = np.random.normal(0, 0.2*stand, size=len(pred[0, :, -1]))
        samples[i, :] = pred[0, :, -1] - random_values

    # 得到每个时间点的置信区间
    lower_bound = np.percentile(samples, 5, axis=0)
    upper_bound = np.percentile(samples, 95, axis=0)

    # 调整错误值，保证置信区间合理性
    # adjusted_lower_bound = lower_bound.copy()
    # adjusted_upper_bound = upper_bound.copy()
    # for i in range(len(lower_bound)):
    #     if lower_bound[i,] > pred[0, i, -1]:
    #         adjusted_lower_bound[i,] = 2 * pred[0, i, -1] - lower_bound[i]
    #     if upper_bound[i,] < pred[0, i, -1]:
    #         adjusted_upper_bound[i,] = 2 * pred[0, i, -1] - upper_bound[i]
    # # adjusted_lower_bound = np.where(lower_bound > pred[0,:,-1], 2 * pred[0,:,-1] - lower_bound, lower_bound)
    # # adjusted_upper_bound = np.where(upper_bound < pred[0,:,-1], 2 * pred[0,:,-1] - upper_bound, upper_bound)

    # lower_bound = adjusted_lower_bound
    # upper_bound = adjusted_upper_bound

    # lower_bound = np.convolve(lower_bound, np.ones(5) / 5, mode='same')
    # upper_bound = np.convolve(upper_bound, np.ones(5) / 5, mode='same')

    return upper_bound, lower_bound


def bootstrap_confidence_interval(preds,model,device,alpha=0.1, n_bootstrap=1000):
    n = len(preds[0,:,-1])
    bootstrapped_intervals = []
    seg = 10
    for _ in range(n_bootstrap):
        seg_intervals = []
        for st in range(0,n,seg):
            end = min(st+seg,n)
            pred_seg = preds[0,st:end,-1]
            boot_idx = np.random.choice([index for index in range(st,end)], replace=True, size=len(pred_seg))
            seg_preds = [preds[0,i,-1] for i in boot_idx]
            seg_intervals.append(seg_preds)
        seg_intervals = [val for sublist in seg_intervals for val in sublist]
        seg_intervals = np.array(seg_intervals).reshape(168,)
        bootstrapped_intervals.append(seg_intervals)
    bootstrapped_intervals = np.array(bootstrapped_intervals)
    # bootstrapped_intervals = []
    # for _ in range(n_bootstrap):
    #     bootstrapped_intervals.append(resample(preds[0,:,-1]))

    lower_bound = np.percentile(bootstrapped_intervals, 100 * alpha / 2, axis=0)
    upper_bound = np.percentile(bootstrapped_intervals, 100 * (1 - alpha / 2), axis=0)

    lower_bound = np.convolve(lower_bound, np.ones(5) / 5, mode='same')
    upper_bound = np.convolve(upper_bound, np.ones(5) / 5, mode='same')
    return upper_bound,lower_bound

def read_data(set_type,df_raw):
    border1s = [24 * 12, 28 * 24 * 12, 30 * 24 * 12]
    border2s = [28 * 24 * 12, 30 * 24 * 12, 31 * 24 * 12]
    border1 = border1s[set_type]
    border2 = border2s[set_type]

    df_data = df_raw[['C1']]
    df = df_data[0:288]
    df_data = np.concatenate((df, df_data), axis=0)

    train_data = df_data[border1s[0]:border2s[0]]
    # scaler = StandardScaler()
    # scaler.fit(train_data.values)
    # data = scaler.transform(df_data.values)
    mean = np.mean(train_data, axis=0)  
    std = np.std(train_data, axis=0)
    data = (df_data - mean) / std

    data_x = data[border1:border2]
    data_y = data[border1:border2]
    data_z = data[border1-288:border2-288]

    seq_x = []
    seq_y = []
    index = 0
    for i in range(int((border2-border1)/288)):
        seq_x.append(np.concatenate((data_z[i*288:i*288+288], data_x[index:index+108]), axis=0))
        seq_y.append(data_y[index+108:index+276])
        index = index+288
    seq_x = np.array(seq_x)
    seq_y = np.array(seq_y)

    return seq_x, seq_y,len(seq_x)

def coverage(up_line,low_line,true):
    covered_count = 0
    total_count = len(true)

    for true_val, upper_val, lower_val in zip(true, up_line, low_line):
        if lower_val <= true_val <= upper_val:
            covered_count += 1

    return covered_count / total_count

def PIS(up_line,low_line,true,r):
    interval = 0
    for i in range(len(up_line)):
        interval = interval+(up_line[i]-low_line[i])
    interval = interval/len(up_line)
    cov = coverage(up_line,low_line,true)
    pis = cov - r*interval
    return pis

def average(X,low,up):
    start_index = next((i for i, x in enumerate(X) if x >= low), None)
    end_index = next((i for i, x in reversed(list(enumerate(X))) if x <= up), None)
    range_sum = sum(X[start_index:end_index + 1])
    range_count = end_index - start_index + 1
    return range_sum / range_count