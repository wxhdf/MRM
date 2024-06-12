# -*- coding: utf-8 -*-
'''
@time: 2021/4/7 15:21

@ author:
'''

import time

import torch
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, f1_score
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[None, :, None] * x + self.bias[None, :, None]
            return x


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def compute_mAP(y_true, y_pred):
    AP = []
    for i in range(len(y_true)):
        AP.append(average_precision_score(y_true[i], y_pred[i]))
    return np.mean(AP)


# def compute_F1(y_true, y_pred):
#     y_true = np.array(y_true)
#     y_pred = np.array(y_pred)
#     total_F1 = 0.0
#     count = 0
#
#     # Ensure predictions are binary (0 or 1)
#     y_pred = (y_pred >= 0.5).astype(int)
#
#     for i in range(len(y_pred)):
#         # Generate confusion matrix for each instance
#         # Note: confusion_matrix returns [[TN, FP], [FN, TP]]
#         tn, fp, fn, tp = confusion_matrix(y_true=y_true[i], y_pred=y_pred[i]).ravel()
#
#         # Calculate precision and recall
#         if (tp + fp) == 0:
#             precision = 0
#         else:
#             precision = tp / (tp + fp)
#
#         if (tp + fn) == 0:
#             recall = 0
#         else:
#             recall = tp / (tp + fn)
#
#         # Calculate F1 score
#         if (precision + recall) == 0:
#             F1 = 0
#         else:
#             F1 = 2 * (precision * recall) / (precision + recall)
#
#         total_F1 += F1
#         count += 1
#
#     # Return average F1 score
#     return total_F1 / count


def compute_F1(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    f1_scores = []
    for i in range(len(y_pred)):
        y_pred[i] = np.where(y_pred[i] >= 0.5, 1, 0)
        f1 = f1_score(y_true=y_true[i], y_pred=y_pred[i])
        f1_scores.append(f1)

    return np.mean(f1_scores)

def compute_ACC(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    total_correct, total = 0, 0
    for i in range(len(y_pred)):
        y_pred[i] = np.where(y_pred[i] >= 0.5, 1, 0)
        tn, fp, fn, tp = confusion_matrix(y_true=y_true[i], y_pred=y_pred[i]).ravel()
        total_correct += tp + tn
        total += tn + fp + fn + tp

    return total_correct / total

# def compute_mAP(y_true, y_pred):
#     y_true = np.array(y_true)
#     y_pred = np.array(y_pred)
#     mAP_sum = 0
#     for i in range(len(y_pred)):
#         y_pred_i = np.where(y_pred[i] >= 0.5, 1, 0)
#         mAP_sum += average_precision_score(y_true[i], y_pred_i)
#     mAP = mAP_sum / len(y_pred)
#     return mAP

def compute_TPR(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sum, count = 0.0, 0
    for i, _ in enumerate(y_pred):
        y_pred[i] = np.where(y_pred[i] >= 0.5, 1, 0)
        (x, y) = confusion_matrix(y_true=y_true[i], y_pred=y_pred[i])[1]
        sum += y / (x + y)
        count += 1

    return sum / count


def compute_AUC(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    class_auc = []
    for i in range(len(y_true[1])):
        class_auc.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
    auc = roc_auc_score(y_true, y_pred)
    return auc, class_auc


# PRINT TIME
def print_time_cost(since):
    time_elapsed = time.time() - since
    return '{:.0f}m{:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)


# KD loss
class KdLoss(nn.Module):
    def __init__(self, alpha, temperature):
        super(KdLoss, self).__init__()
        self.alpha = alpha
        self.T = temperature

    def forward(self, outputs, labels, teacher_outputs):
        kd_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs / self.T, dim=1),
                                                      F.softmax(teacher_outputs / self.T, dim=1)) * (
                          self.alpha * self.T * self.T) + F.binary_cross_entropy_with_logits(outputs, labels) * (
                          1. - self.alpha)
        return kd_loss

def merge_pre_bn(module, pre_bn_1, pre_bn_2=None):
    """ Merge pre BN to reduce inference runtime.
    """
    weight = module.weight.data
    if module.bias is None:
        zeros = torch.zeros(module.out_channels, device=weight.device).type(weight.type())
        module.bias = nn.Parameter(zeros)
    bias = module.bias.data
    if pre_bn_2 is None:
        assert pre_bn_1.track_running_stats is True, "Unsupport bn_module.track_running_stats is False"
        assert pre_bn_1.affine is True, "Unsupport bn_module.affine is False"

        scale_invstd = pre_bn_1.running_var.add(pre_bn_1.eps).pow(-0.5)
        extra_weight = scale_invstd * pre_bn_1.weight
        extra_bias = pre_bn_1.bias - pre_bn_1.weight * pre_bn_1.running_mean * scale_invstd
    else:
        assert pre_bn_1.track_running_stats is True, "Unsupport bn_module.track_running_stats is False"
        assert pre_bn_1.affine is True, "Unsupport bn_module.affine is False"

        assert pre_bn_2.track_running_stats is True, "Unsupport bn_module.track_running_stats is False"
        assert pre_bn_2.affine is True, "Unsupport bn_module.affine is False"

        scale_invstd_1 = pre_bn_1.running_var.add(pre_bn_1.eps).pow(-0.5)
        scale_invstd_2 = pre_bn_2.running_var.add(pre_bn_2.eps).pow(-0.5)

        extra_weight = scale_invstd_1 * pre_bn_1.weight * scale_invstd_2 * pre_bn_2.weight
        extra_bias = scale_invstd_2 * pre_bn_2.weight *(pre_bn_1.bias - pre_bn_1.weight * pre_bn_1.running_mean * scale_invstd_1 - pre_bn_2.running_mean) + pre_bn_2.bias

    if isinstance(module, nn.Linear):
        extra_bias = weight @ extra_bias
        weight.mul_(extra_weight.view(1, weight.size(1)).expand_as(weight))
    elif isinstance(module, nn.Conv2d):
        assert weight.shape[2] == 1 and weight.shape[3] == 1
        weight = weight.reshape(weight.shape[0], weight.shape[1])
        extra_bias = weight @ extra_bias
        weight.mul_(extra_weight.view(1, weight.size(1)).expand_as(weight))
        weight = weight.reshape(weight.shape[0], weight.shape[1], 1, 1)
    bias.add_(extra_bias)

    module.weight.data = weight
    module.bias.data = bias
