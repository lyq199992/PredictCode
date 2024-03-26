import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import math

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


# def visual(true, up_preds, low_preds, preds=None, name='./pic/test.pdf'):
#     """
#     Results visualization
#     """
#     plt.figure()
#     plt.plot(true, label='GroundTruth', linewidth=2)
#     if preds is not None:
#         plt.plot(up_preds, label='upper_bound', linewidth=1)
#         plt.plot(low_preds, label='lower_bound', linewidth=1)
#         plt.plot(preds, label='Prediction', linewidth=2)
#     plt.legend()
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


def Conference_interval(true, pred, conf, mse, Acc):
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
        # random_values = np.random.normal(0, residuals, size=len(pred[0,:,-1]))
        # samples[i, :] = pred[0,:,-1] + random_values
        # random_values = np.random.normal(0, 0.4*np.std(pred[0,:,-1]), size=len(pred[0,:,-1]))
        random_values = np.random.normal(0, 0.2 * stand, size=len(pred[0, :, -1]))
        samples[i, :] = pred[0, :, -1] + random_values

    # 得到每个时间点的置信区间
    lower_bound = np.percentile(samples, 10, axis=0)
    upper_bound = np.percentile(samples, 90, axis=0)

    # 调整错误值，保证置信区间合理性
    adjusted_lower_bound = lower_bound.copy()
    adjusted_upper_bound = upper_bound.copy()
    for i in range(len(lower_bound)):
        if lower_bound[i,] > pred[0, i, -1]:
            adjusted_lower_bound[i,] = 2 * pred[0, i, -1] - lower_bound[i]
        if upper_bound[i,] < pred[0, i, -1]:
            adjusted_upper_bound[i,] = 2 * pred[0, i, -1] - upper_bound[i]
    # adjusted_lower_bound = np.where(lower_bound > pred[0,:,-1], 2 * pred[0,:,-1] - lower_bound, lower_bound)
    # adjusted_upper_bound = np.where(upper_bound < pred[0,:,-1], 2 * pred[0,:,-1] - upper_bound, upper_bound)

    lower_bound = adjusted_lower_bound
    upper_bound = adjusted_upper_bound

    print(Acc)

    # 通过Acc来调整上下界
    for j in range(len(pred[0, :, -1])):
        up_wrong = upper_bound[j,] - pred[0, j, -1]
        low_wrong = pred[0, j, -1] - lower_bound[j,]
        if Acc[j] < 0.5:
            if pred[0, j, -1] < true[0, j, -1]:
                up_wrong = abs(pred[0, j, -1]) * Acc[j]
                upper_bound[j,] = upper_bound[j,] + up_wrong
            else:
                low_wrong = abs(pred[0, j, -1]) * Acc[j]
                lower_bound[j,] = lower_bound[j,] - low_wrong

    lower_bound = np.convolve(lower_bound, np.ones(5) / 5, mode='same')
    upper_bound = np.convolve(upper_bound, np.ones(5) / 5, mode='same')

    return upper_bound, lower_bound


def Bootstrap_conference(true, pred, conf, mse):
    # 定义Bootstrap采样次数
    num_bootstrap_samples = 1000

    # 初始化数组以保存Bootstrap采样结果
    bootstrap_samples = np.zeros((num_bootstrap_samples,))

    # 进行Bootstrap采样
    for i in range(num_bootstrap_samples):
        # 从观测数据中有放回地进行抽样
        bootstrap_sample = np.random.choice(pred[0, :, -1], size=len(pred[0, :, -1]), replace=True)
        # 计算采样的均值，并存储到数组中
        bootstrap_samples[i] = np.mean(bootstrap_sample)

    # 得到均值的置信区间
    lower_bound = np.percentile(bootstrap_samples, 10)
    upper_bound = np.percentile(bootstrap_samples, 90)

    print(lower_bound.shape, upper_bound.shape)
    exit()
    return upper_bound, lower_bound