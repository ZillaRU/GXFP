import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def eval_threshold(labels_all, preds_all):
    for i in range(int(0.5 * len(labels_all))):
        if preds_all[2*i] > 0.95 and preds_all[2*i+1] > 0.95:
            preds_all[2*i] = max(preds_all[2*i], preds_all[2*i+1])
            preds_all[2*i+1] = preds_all[2*i]
        else:
            preds_all[2*i] = min(preds_all[2*i], preds_all[2*i+1])
            preds_all[2*i+1] = preds_all[2*i]
    fpr, tpr, thresholds = roc_curve(labels_all, preds_all)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    preds_all_ = []
    for p in preds_all:
        if p >=optimal_threshold:
            preds_all_.append(1)
        else:
            preds_all_.append(0)
    return preds_all, preds_all_


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss