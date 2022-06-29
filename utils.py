import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def sinkhorn(pred, eta, rec=False):
    PS = pred.detach()
    K = PS.shape[1]
    N = PS.shape[0]
    PS = PS.T
    c = torch.ones((N, 1)) / N
    r = torch.ones((K, 1)) / K
    r = r.cuda()
    c = c.cuda()
    # average column mean 1/N
    PS = torch.pow(PS, eta)  # K x N
    r_init = r
    inv_N = 1. / N
    err = 1e6
    # error rate
    _counter = 1
    for i in range(50):
        if err < 1e-1:
            break
        r = r_init * (1 / (PS @ c))  # (KxN)@(N,1) = K x 1
        # 1/K(Plambda * beta)
        c_new = inv_N / (r.T @ PS).T  # ((1,K)@(KxN)).t() = N x 1
        # 1/N(alpha * Plambda)
        if _counter % 10 == 0:
            err = torch.sum(c_new) + torch.sum(r)
            if torch.isnan(err):
                # This may very rarely occur (maybe 1 in 1k epochs)
                # So we do not terminate it, but return a relaxed solution
                #print('====> Nan detected, return relaxed solution')
                pred_new = pred + 1e-5 * (pred == 0)
                relaxed_PS, _ = sinkhorn(pred_new, eta, rec=True)
                z = (1.0 * (pred != 0))
                relaxed_PS = relaxed_PS * z
                return relaxed_PS, True
        c = c_new
        _counter += 1
    PS *= torch.squeeze(c)
    PS = PS.T
    PS *= torch.squeeze(r)
    PS *= N
    return PS.detach(), False


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

class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

def linear_rampup2(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr
    eta_min = lr * (args.lr_decay_rate ** 3)
    if (epoch < 420):
        lr = 8 * eta_min + (lr - 8 * eta_min) * (1 + math.cos(math.pi * epoch / 420)) / 2
    else:
        lr = eta_min  + (8 * eta_min - eta_min) * (1 + math.cos(math.pi * (epoch-420) / (args.num_epochs-420))) / 2

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate_420(args, optimizer, epoch):
    lr = args.lr
    eta_min = lr * (args.lr_decay_rate ** 3)
    if (epoch < 420):
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / 420)) / 2
    else:
        lr = eta_min

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate_300(args, optimizer, epoch):
    lr = args.lr
    eta_min = lr * (args.lr_decay_rate ** 3)
    if (epoch < 300):
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / 300)) / 2
    else:
        lr = eta_min * 2

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate_ori(args, optimizer, epoch):
    lr = args.lr
    eta_min = lr * (args.lr_decay_rate ** 3)
    if (epoch < args.num_epochs):
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / args.num_epochs)) / 2
    else:
        lr = eta_min

def adjust_learning_rate_multistep(args, optimizer, epoch,milestones):
    lr = args.lr * (0.1 ** len([m for m in milestones if m < epoch]))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate_linear(args, optimizer, epoch):
    lr = args.lr
    eta_min = lr * (args.lr_decay_rate ** 3)
    lr = lr + (eta_min - lr) * (epoch / args.num_epochs)
class SupConLoss(nn.Module):
    """Following Supervised Contrastive Learning: 
        https://arxiv.org/pdf/2004.11362.pdf."""
    def __init__(self, args,temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.register_buffer("queue", torch.randn(8192, 128))
        self.register_buffer("queue_pseudo", torch.randn(8192, args.num_class))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.moco_queue = 8192

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        # gather keys before updating queue
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.moco_queue % batch_size == 0  # for simplicity
        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size, :] = keys
        self.queue_pseudo[ptr:ptr + batch_size, :] = labels
        ptr = (ptr + batch_size) % 8192  # move pointer
        self.queue_ptr[0] = ptr



    def forward(self, features, mask=None, batch_size=-1):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        # SupCon loss (Partial Label Mode)
        mask = mask.float().detach().to(device)
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features[:batch_size], features.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
    
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        
        return loss