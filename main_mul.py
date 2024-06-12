# -*- coding: utf-8 -*-
"""
@time: 2021/4/15 15:40

@ author:
"""
import warnings

import torch.nn.functional as F
import torch, time, os
from torch.optim.lr_scheduler import MultiStepLR
# from model import ECGViT,M111
# from M1 import M111
# from model import M1
# from model2 import M1
import utils
from torch import nn, optim
from dataset import load_datasets
from config import config
from sklearn.metrics import roc_auc_score
import numpy as np
import random
from models import model2

# from model250 import Fuse

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DistillKL(nn.Module):
    """KL divergence for distillation"""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / y_s.shape[0]
        return loss


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_checkpoint(filepath, model, optimizer):
    # 确保checkpoint文件存在
    if os.path.isfile(filepath):
        print("Loading checkpoint '{}'".format(filepath))
        checkpoint = torch.load(filepath)
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return model
    else:
        print("No checkpoint found at '{}'".format(filepath))
        return None


def save_checkpoint(best_auc, model, optimizer, epoch):
    print('Model Saving...')
    if config.device_num > 1:
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    torch.save({
        'model_state_dict': model_state_dict,
        'global_epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'best_auc': best_auc,
    }, os.path.join('checkpoints', config.model_name + '_' + config.experiment + '_checkpoint_best.pth'))


def train_epoch(model, optimizer, criterion, criterion_Cross, scheduler_teacher_model, train_dataloader):
    model.train()
    loss_meter, it_count = 0, 0
    m_loss = 0
    outputs1 = []
    outputs2 = []
    outputs3 = []
    outputs = []
    targets = []
    for inputs1, target in train_dataloader:
        # inputs1 = inputs1 + torch.randn_like(inputs1) * 0.1
        inputs1 = inputs1.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        feat_ins, out_ins, feat_all, out = model(inputs1)
        feat_ins, out_ins, feat_all, out = feat_ins.to(torch.float), out_ins.to(torch.float), feat_all.to(
            torch.float), out.to(torch.float)
        loss1 = criterion(out_ins, target)
        loss2 = criterion(out, target)
        mutual_loss1 = criterion_Cross(feat_ins, feat_all) + criterion_Cross(feat_all, feat_ins)
        mutual_loss2 = criterion_Cross(out_ins, out) + criterion_Cross(out, out_ins)
        mutual_loss = mutual_loss1 * 0.01 + mutual_loss2 * 0.1
        # torch.autograd.set_detect_anomaly(True) + mutual_loss * 0.1
        loss = loss1 + loss2 + mutual_loss
        m_loss += mutual_loss
        loss.backward()
        optimizer.step()
        loss_meter += loss.item()
        it_count += 1
        out1 = torch.sigmoid(out_ins)
        out = torch.sigmoid(out)
        for i in range(len(target)):
            outputs1.append(out1[i].cpu().detach().numpy())
            outputs.append(out[i].cpu().detach().numpy())
            targets.append(target[i].cpu().detach().numpy())
    # scheduler_teacher_model.step()
    auc1 = roc_auc_score(targets, outputs1)
    TPR1 = utils.compute_TPR(targets, outputs1)
    acc1 = utils.compute_ACC(targets, outputs1)
    f11 = utils.compute_F1(targets, outputs1)
    auc = roc_auc_score(targets, outputs)
    TPR = utils.compute_TPR(targets, outputs)
    acc = utils.compute_ACC(targets, outputs)
    f1 = utils.compute_F1(targets, outputs)
    print('train_loss1: %.4f, m_loss1: %.4f, auc: %.4f,  TPR: %.4f, acc: %.4f, f1: %.4f' % (
        loss_meter / it_count, m_loss / it_count, auc1, TPR1, acc1, f11))
    print('train_loss: %.4f, auc: %.4f,  TPR: %.4f, acc: %.4f, f1: %.4f' % (
        loss_meter / it_count, auc, TPR, acc, f1))
    print("-----------------------------------------------------------------------------")
    return 0


# val and test
def val_epoch(model, criterion, criterion_Cross, val_dataloader):
    model.eval()
    loss_meter, it_count = 0, 0
    outputs1 = []
    outputs2 = []
    outputs3 = []
    outputs = []
    targets = []
    with torch.no_grad():
        for inputs1, target in val_dataloader:

            # inputs1 = inputs1 + torch.randn_like(inputs1) * 0.1
            inputs1 = inputs1.to(device)
            target = target.to(device)
            feat_ins, out_ins, feat_all, out = model(inputs1)
            feat_ins, out_ins, feat_all, out = feat_ins.to(torch.float), out_ins.to(torch.float), feat_all.to(
                torch.float), out.to(torch.float)
            loss1 = criterion(out_ins, target)
            loss2 = criterion(out, target)
            # mutual_loss = criterion_Cross(feat_ins, feat_all) + criterion_Cross(feat_all, feat_ins)
            loss = loss1 + loss2
            loss_meter += loss.item()
            it_count += 1
            out1 = torch.sigmoid(out_ins)
            out = torch.sigmoid(out)
            for i in range(len(target)):
                outputs1.append(out1[i].cpu().detach().numpy())
                outputs.append(out[i].cpu().detach().numpy())
                targets.append(target[i].cpu().detach().numpy())
            # scheduler_teacher_model.step()
        auc1 = roc_auc_score(targets, outputs1)
        TPR1 = utils.compute_TPR(targets, outputs1)
        acc1 = utils.compute_ACC(targets, outputs1)
        f11 = utils.compute_F1(targets, outputs1)
        auc = roc_auc_score(targets, outputs)
        TPR = utils.compute_TPR(targets, outputs)
        acc = utils.compute_ACC(targets, outputs)
        f1 = utils.compute_F1(targets, outputs)
    print('val_loss1: %.4f,   val_macro_auc: %.4f,  TPR: %.4f, acc: %.4f, f1: %.4f' % (
        loss_meter / it_count, auc1, TPR1, acc1, f11))
    print('val_loss: %.4f, macro_auc: %.4f,  TPR: %.4f, acc: %.4f, f1: %.4f' % (
        loss_meter / it_count, auc, TPR, acc, f1))
    print("-----------------------------------------------------------------------------")
    return 0


def test_epoch(model, criterion, criterion_Cross, val_dataloader):
    model.eval()
    loss_meter, it_count = 0, 0
    m_loss = 0
    outputs1 = []
    outputs2 = []
    outputs3 = []
    outputs = []
    targets = []
    with torch.no_grad():
        for inputs1, target in val_dataloader:
            # inputs1 = inputs1 + torch.randn_like(inputs1) * 0.1
            inputs1 = inputs1.to(device)
            target = target.to(device)
            feat_ins, out_ins, feat_all, out = model(inputs1)
            feat_ins, out_ins, feat_all, out = feat_ins.to(torch.float), out_ins.to(torch.float), feat_all.to(
                torch.float), out.to(torch.float)
            loss1 = criterion(out_ins, target)
            loss2 = criterion(out, target)
            mutual_loss = criterion_Cross(feat_ins, feat_all) + criterion_Cross(feat_all, feat_ins)
            loss = loss1 * 0.75 + loss2 + 0.5 * mutual_loss
            loss_meter += loss.item()
            # m_loss += mutual_loss.item()/
            it_count += 1
            out1 = torch.sigmoid(out_ins)
            out = torch.sigmoid(out)
            for i in range(len(target)):
                outputs1.append(out1[i].cpu().detach().numpy())
                outputs.append(out[i].cpu().detach().numpy())
                targets.append(target[i].cpu().detach().numpy())
            # scheduler_teacher_model.step()
        auc1 = roc_auc_score(targets, outputs1)
        TPR1 = utils.compute_TPR(targets, outputs1)
        acc1 = utils.compute_ACC(targets, outputs1)
        f11 = utils.compute_F1(targets, outputs1)
        auc = roc_auc_score(targets, outputs)
        TPR = utils.compute_TPR(targets, outputs)
        acc = utils.compute_ACC(targets, outputs)
        f1 = utils.compute_F1(targets, outputs)
    print('test_loss1: %.4f, m_loss1: %.4f, test_macro_auc: %.4f,  TPR: %.4f, acc: %.4f, f1: %.4f' % (
        loss_meter / it_count, m_loss / it_count, auc1, TPR1, acc1, f11))
    print('test_loss: %.4f, macro_auc: %.4f,  TPR: %.4f, acc: %.4f, f1: %.4f' % (
        loss_meter / it_count, auc, TPR, acc, f1))
    return auc1, TPR1, acc1, f11, auc, TPR, acc, f1


def train(i, config=config):
    # seed
    setup_seed(config.seed)
    print('torch.cuda.is_available:', torch.cuda.is_available())

    # datasets
    train_dataloader, val_dataloader, test_dataloader, num_classes = load_datasets(experiment=config.experiment,
                                                                                   datafolder='../data/ptbxl/')
    from models import duibi
    model = duibi.Dui1(num_classes=num_classes)

    # from models import duibi, duibi2
    # if i == 0:
    #     model = duibi.Dui1(num_classes=num_classes)
    # else:
    #     model = duibi2.Dui2(num_classes=num_classes)
    print('model_name:{}, num_classes={}'.format(config.model_name, num_classes))
    model = model.to(device)

    # optimizer and loss SGD
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    # optimizer = optim.SGD(model.parameters(), lr=config.lr)
    scheduler_teacher_model = MultiStepLR(optimizer, milestones=[50, 100, 120], gamma=0.1)

    criterion = nn.BCEWithLogitsLoss()
    criterion_Cross = DistillKL(T=4.0)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    # =========>train<=========
    bauc1, btpr1, bacc1, bf11 = 0, 0, 0, 0
    bauc2, btpr2, bacc2, bf12 = 0, 0, 0, 0
    for epoch in range(1, config.max_epoch + 1):
        print('#epoch: {}  batch_size: {}  Current Learning Rate: {}'.format(epoch, config.batch_size,
                                                                             config.lr))

        since = time.time()
        train_epoch(model, optimizer, criterion, criterion_Cross, scheduler_teacher_model,
                    train_dataloader)

        val_epoch(model, criterion, criterion_Cross, val_dataloader)
        auc1, TPR1, acc1, f11, auc, TPR, acc, f1 = test_epoch(model, criterion, criterion_Cross, test_dataloader)
        if max(bauc1 + bf11, bauc2 + bf12) < max(auc1 + f11, auc + f1):
            bauc1, btpr1, bacc1, bf11 = auc1, TPR1, acc1, f11
            bauc2, btpr2, bacc2, bf12 = auc, TPR, acc, f1
            save_checkpoint(bauc1, model, optimizer, epoch)
        if epoch == config.max_epoch:
            print('btest_macro_auc: %.4f,  TPR: %.4f, acc: %.4f, f1: %.4f' % (bauc1, btpr1, bacc1, bf11))
            print('btest_macro_auc: %.4f,  TPR: %.4f, acc: %.4f, f1: %.4f' % (bauc2, btpr2, bacc2, bf12))

        # if epoch == config.max_epoch:
        #     path = os.path.join('checkpoints', config.model_name + '_' + config.experiment + '_checkpoint_best.pth')
        #     load_model = load_checkpoint(path, model, optimizer)
        #     test_loss, test_auc, test_TPR = test_epoch(load_model, criterion, test_dataloader)
        #     result_list = [[epoch, train_loss, train_auc, train_TPR,
        #                     val_loss, val_auc, val_TPR,
        #                     test_loss, test_auc, test_TPR]]
        #     if epoch == 1:
        #         columns = ['epoch', 'train_loss', 'train_auc', 'train_TPR',
        #                    'val_loss', 'val_auc', 'val_TPR',
        #                    'test_loss', 'test_auc', 'test_TPR']
        #     else:
        #         columns = ['', '', '', '', '', '', '', '', '', '']
        #     dt = pd.DataFrame(result_list, columns=columns)
        #     dt.to_csv(config.model_name + config.experiment + 'result.csv', mode='a')

        print('time:%s\n' % (utils.print_time_cost(since)))


if __name__ == '__main__':
    # train()'exp0', 'exp1','exp0', 'exp1', 'exp1.1', 'exp1.1.1',
    config.seed = 1
    for i in range(2):
        config.experiment = 'exp0'
        train(i, config)
    # for exp in ['exp0', 'exp1', 'exp1.1', 'exp1.1.1', 'exp2', 'exp3']:
    #     # if exp == 'exp0':
    #     #     config.seed = 1
    #     # elif exp == 'exp1':
    #     #     config.seed = 1
    #     # elif exp == 'exp1.1':
    #     #     config.seed = 1
    #     # elif exp == 'exp1.1.1':
    #     #     config.seed = 1
    #     # elif exp == 'exp2':
    #     #     config.seed = 1
    #     # elif exp == 'exp3':
    #     #     config.seed = 1
    #     # exp = 'exp1'
    #     config.experiment = exp
    #     train(config)

    # config.datafolder = '../data/CPSC/'
    # config.experiment = 'ptbxl'
    # # config.seed = 7
    # train(config)

    # config.datafolder = '../data/hf/'
    # config.experiment = 'hf'
    # config.seed = 9
    # train(config)
