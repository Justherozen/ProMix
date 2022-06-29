import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CE_Soft_Label(nn.Module):
    def __init__(self):
        super().__init__()
        # print('Calculating uniform targets...')
        # calculate confidence
        self.confidence = None
        self.gamma = 2.0
        self.alpha = 0.25
    def init_confidence(self, noisy_labels, num_class):
        noisy_labels = torch.Tensor(noisy_labels).long().cuda()
        self.confidence = F.one_hot(noisy_labels, num_class).float().clone().detach()

    def forward(self, outputs, targets=None):
        logsm_outputs = F.log_softmax(outputs, dim=1)
        final_outputs = logsm_outputs * targets.detach()
        loss_vec = - ((final_outputs).sum(dim=1))
        #p = torch.exp(-loss_vec)
        #loss_vec =  (1 - p) ** self.gamma * loss_vec
        average_loss = loss_vec.mean()
        return loss_vec

    @torch.no_grad()
    def confidence_update(self, temp_un_conf, batch_index, conf_ema_m):
        with torch.no_grad():
            _, prot_pred = temp_un_conf.max(dim=1)
            pseudo_label = F.one_hot(prot_pred, temp_un_conf.shape[1]).float().cuda().detach()
            self.confidence[batch_index, :] = conf_ema_m * self.confidence[batch_index, :]\
                 + (1 - conf_ema_m) * pseudo_label
        return None

def linear_rampup2(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / args.num_epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
