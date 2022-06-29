from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from PreResNet import *
# from wideresnet import SupConWideResNet
from sklearn.mixture import GaussianMixture
import dataloader_cifarn as dataloader
from utils import *
from fmix import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('-lr_decay_rate', type=float, default=0.1,
                    help='decay rate for learning rate')
parser.add_argument('-lr_decay_epochs', type=str, default='100,200,300',
                    help='where to decay lr, can be a list')
parser.add_argument('--cosine', action='store_true', default=False,
                    help='use cosine lr schedule')
parser.add_argument('--noise_type', type=str, help='clean, aggre, worst, rand1, rand2, rand3, clean100, noisy100',
                    default='clean')
parser.add_argument('--noise_path', type=str, help='path of CIFAR-10_human.pt', default=None)
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--proto_m', default=0.9, type=float, help='speed of prototype updating')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=40, type=int)
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=100, type=int)
parser.add_argument('--data_path', default=None, type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--is_human', action='store_true', default=False)
parser.add_argument('--rho_range', default='0.2,0.6', type=str,
                    help='ratio of clean labels (rho)')
parser.add_argument('--tau', default=0.99, type=float,
                    help='high-confidence selection threshold')
parser.add_argument('--pretrain_ep', default=0, type=int)
parser.add_argument('--warmup_ep', default=50, type=int)
parser.add_argument('--topk', default=4, type=int)
parser.add_argument('--unrel_pseudo', default='sharpen', type=str)
parser.add_argument('--low_conf_del', action='store_true', default=False)
parser.add_argument('--threshold', default=0.95, type=float)


args = parser.parse_args()
[args.rho_start, args.rho_end] = [float(item) for item in args.rho_range.split(',')]
iterations = args.lr_decay_epochs.split(',')
args.lr_decay_epochs = list([])
for it in iterations:
    args.lr_decay_epochs.append(int(it))
print(args)

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# Hyper Parameters
noise_type_map = {'clean': 'clean_label', 'worst': 'worse_label', 'aggre': 'aggre_label', 'rand1': 'random_label1',
                  'rand2': 'random_label2', 'rand3': 'random_label3', 'clean100': 'clean_label',
                  'noisy100': 'noisy_label'}
args.noise_type = noise_type_map[args.noise_type]
# load dataset
if args.data_path is None:
    if args.dataset == 'cifar10':
        args.data_path = './cifar-10'
    elif args.dataset == 'cifar100':
        args.data_path = './cifar-100'
    else:
        # raise NameError(f'Undefined dataset {args.dataset}')
        pass

if args.noise_path is None:
    if args.dataset == 'cifar10':
        args.noise_path = './data/CIFAR-10_human.pt'
    elif args.dataset == 'cifar100':
        args.noise_path = './data/CIFAR-100_human.pt'
    else:
        # raise NameError(f'Undefined dataset {args.dataset}')
        pass

def selection(batch_size, rho, pseudo_loss_vec, pred, pseudo_labels):
    idx_chosen_sm = []
    weights_sl = torch.zeros(pred.shape[0]).cuda().detach()
    weights_hc = torch.zeros(pred.shape[0]).cuda().detach()
    # small loss selection
    # idx = pseudo_loss_vec.sort()[1]
    # idx_chosen_sm = idx[:int(batch_size* rho)]
    # weights_sl[idx_chosen_sm] = 1

    pseudo_label_idx = pseudo_labels.max(dim=1)[1]
    idx_chosen_sm = []
    for j in range(pred.shape[1]):
        indices = np.where(pseudo_label_idx.cpu().numpy()==j)[0]
        # torch.where will cause device error
        if len(indices) == 0:
            continue
        bs_j = batch_size * (1. / pred.shape[1])
        pseudo_loss_vec_j = pseudo_loss_vec[indices]
        sorted_idx_j = pseudo_loss_vec_j.sort()[1].cpu().numpy()
        partition_j = max(min(int(math.ceil(bs_j*rho)), len(indices)), 1)
        # at least one example
        idx_chosen_sm.append(indices[sorted_idx_j[:partition_j]])
    idx_chosen_sm = np.concatenate(idx_chosen_sm)
    weights_sl[idx_chosen_sm] = 1

    # high confidence selection
    # high_conf_cond = pred.max(dim=1)[0] > args.tau
    high_conf_cond = (pseudo_labels * pred).sum(dim=1) > args.tau
    weights_hc[high_conf_cond] = 1
    # weights_hc_large = 1 * ((weights_hc - weights_sl) == 1)
    # hc_not_sl_cond = weights_hc_large == 1
    # pred_pseudo_labels = pred.max(dim=1)[1]
    # pseudo_labels[hc_not_sl_cond] = F.one_hot(pred_pseudo_labels[hc_not_sl_cond], pred.shape[1]).float()
    # change the pseudo-labels to prediction itself on high-conf data
    weights = ((weights_hc + weights_sl) > 0) * 1

    idx_chosen = torch.where(weights == 1)[0]
    idx_unchosen = torch.where(weights == 0)[0]
    return idx_chosen, idx_unchosen, weights

# Training
def train(epoch, net, net2, optimizer, labeled_trainloader, unlabeled_trainloader, prototypes):
    net.train()
    net2.train()  # fix one network and train the other
    
    rho = args.rho_start + (args.rho_end - args.rho_start) * linear_rampup2(epoch, args.warmup_ep)
    w = linear_rampup2(epoch, args.warmup_ep)
    # beta = 0.1 + 0.9 * linear_rampup2(epoch, 800)
    beta = 0.1

    # args.tau = 0.99 if epoch < args.num_epochs - 100 else 0.98

    pseudo_filtered_list = []
    pseudo_labels_list = []
    trues_list = []
    unrel_pseudo_list = []

    num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x, true_labels, index) in enumerate(labeled_trainloader):
        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1, 1), 1)
        w_x = w_x.view(-1, 1).type(torch.FloatTensor)

        index = index.cuda()
        inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
        outputs_x = net(inputs_x)
        outputs_x2 = net(inputs_x2)
        outputs_a = net2(inputs_x)
        outputs_a2 = net2(inputs_x2)
        # net1 for classifier, net2 for contrastive learning
        
        with torch.no_grad():
            # label refinement of labeled samples
            px = torch.softmax(outputs_x, dim=1)
            px2 = torch.softmax(outputs_a, dim=1)
            pred_net2 = F.one_hot(px2.max(dim=1)[1], args.num_class).float()

            targets_x = labels_x.clone().detach()

            # pseudo_loss_vec_l = CEsoft(outputs_x, targets=targets_x)
            # idx_chosen, idx_unchosen, weights = selection(batch_size, rho, pseudo_loss_vec_l, px, targets_x)

            high_conf_cond = (labels_x * px).sum(dim=1) > args.tau
            w_x[high_conf_cond] = 1
            idx_chosen = torch.where(w_x == 1)[0]
            idx_unchosen = torch.where(w_x == 0)[0]
            
            pseudo_label_l = targets_x * w_x + pred_net2 * (1 - w_x)

            w_x2 = w_x.clone()
            if (1. * idx_chosen.shape[0] / batch_size) < args.threshold:
                # when clean data is insufficient, try to incorporate more examples
                score1 = (pseudo_label_l * px).sum(dim=1)
                score2 = (pseudo_label_l * px2).sum(dim=1)
                high_conf_cond2 = (score1 > args.tau) * (score2 > args.tau)
                # both nets agrees
                high_conf_cond2 = (1. * high_conf_cond2 - w_x.squeeze()) > 0
                # remove already selected examples; newly selected
                hc2_idx = torch.where(high_conf_cond2)[0]

                max_to_sel_num = int(batch_size * args.threshold) - idx_chosen.shape[0]
                # maximally select batch_size * args.threshold; idx_chosen.shape[0] select already
                if high_conf_cond2.sum() > max_to_sel_num:
                    # to many examples selected, remove some low conf examples
                    score_mean = ((pseudo_label_l * px2).sum(dim=1) + (pseudo_label_l * px).sum(dim=1)) / 2
                    idx_remove = (-score_mean[hc2_idx]).sort()[1][max_to_sel_num:]
                    # take top scores
                    high_conf_cond2[hc2_idx[idx_remove]] = False
                w_x2[high_conf_cond2] = 1
            idx_chosen_new = torch.where(w_x2 == 1)[0]
            # idx_unchosen_new = torch.where(w_x2 == 0)[0]

            trues_list.append(true_labels)
            filters = torch.zeros(true_labels.shape[0])
            filters[idx_chosen_new.cpu()] = 1
            pseudo_filtered_list.append(filters)
            pseudo_labels_list.append(pseudo_label_l.cpu())

        l = np.random.beta(4, 4)
        l = max(l, 1-l)
        X_w_c = inputs_x[idx_chosen_new]
        pseudo_label_c = pseudo_label_l[idx_chosen_new]
        idx = torch.randperm(X_w_c.size(0))
        X_w_c_rand = X_w_c[idx]
        pseudo_label_c_rand = pseudo_label_c[idx]
        X_w_c_mix = l * X_w_c + (1 - l) * X_w_c_rand        
        pseudo_label_c_mix = l * pseudo_label_c + (1 - l) * pseudo_label_c_rand
        logits_mix = net(X_w_c_mix)
        loss_mix = CEsoft(logits_mix, targets=pseudo_label_c_mix).mean()
        # mixup loss
        x_fmix = fmix(X_w_c)
        logits_fmix = net(x_fmix)
        loss_fmix = fmix.loss(logits_fmix, (pseudo_label_c.detach()).long())
        # fmixup loss
        
        loss_cr = CEsoft(outputs_x2[idx_chosen_new], targets=pseudo_label_l[idx_chosen_new]).mean()
        # consistency loss
        
        loss_ce = CEsoft(outputs_x[idx_chosen_new], targets=pseudo_label_l[idx_chosen_new]).mean()
        # above: loss for reliable samples

        # ptx = px ** (1 / args.T)
        # ptx = ptx / ptx.sum(dim=1, keepdim=True)
        # targets_urel = ptx.detach()
        # # unrel_pseudo_list.append(targets_urel.cpu())

        # loss_urel = CEsoft(outputs_x[idx_unchosen_new], targets=targets_urel[idx_unchosen_new]).mean()\
        #           + w * CEsoft(outputs_x2[idx_unchosen_new], targets=targets_urel[idx_unchosen_new]).mean()
        # # consistency loss on unreliable examples
        # # above: loss for unreliable samples
        
        loss_net1 = loss_ce + w * (loss_cr + loss_mix + loss_fmix)

        #  -------  loss for net2
        # pred_net2 = F.one_hot(px2.max(dim=1)[1], args.num_class).float()
        # pseudo_label_l2 = targets_x * w_x + pred_net2 * (1 - w_x)
        pseudo_label_l2 = pseudo_label_l
        unrel_pseudo_list.append(px2.cpu())

        high_conf_cond2 = px2.max(dim=1)[0] > args.tau
        w_x[high_conf_cond2] = 1
        idx_chosen = torch.where(w_x == 1)[0]
        idx_unchosen = torch.where(w_x == 0)[0]

        # idx_chosen = idx_chosen_new
        # idx_unchosen = idx_unchosen_new

        l = np.random.beta(4, 4)
        l = max(l, 1-l)
        X_w_c = inputs_x[idx_chosen]
        pseudo_label_c = pseudo_label_l2[idx_chosen]
        idx = torch.randperm(X_w_c.size(0))
        X_w_c_rand = X_w_c[idx]
        pseudo_label_c_rand = pseudo_label_c[idx]
        X_w_c_mix2 = l * X_w_c + (1 - l) * X_w_c_rand        
        pseudo_label_c_mix2 = l * pseudo_label_c + (1 - l) * pseudo_label_c_rand
        logits_mix2 = net2(X_w_c_mix2)
        loss_mix2 = CEsoft(logits_mix2, targets=pseudo_label_c_mix2).mean()
        # mixup loss
        x_fmix2 = fmix(X_w_c)
        logits_fmix2 = net2(x_fmix2)
        loss_fmix2 = fmix.loss(logits_fmix2, (pseudo_label_c.detach()).long())
        # fmixup loss

        loss_cr2 = CEsoft(outputs_a2[idx_chosen], targets=pseudo_label_l2[idx_chosen]).mean()
        # consistency loss
        
        loss_ce2 = CEsoft(outputs_a[idx_chosen], targets=pseudo_label_l2[idx_chosen]).mean()

        loss_net2 = loss_ce2 + w * (loss_cr2 + loss_mix2 + loss_fmix2)
        # the loss of the contrastive branch

        loss = loss_net1 + loss_net2
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0 :
            print('%s:%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Net1 loss: %.2f  Net2 loss: %.2f'
                         % (args.dataset, args.noise_type, epoch, args.num_epochs, batch_idx + 1, num_iter,
                            loss_net1.item(), loss_net2.item()))

    pseudo_labels_list = torch.cat(pseudo_labels_list, dim=0)
    pseudo_filtered_list = torch.cat(pseudo_filtered_list, dim=0)
    trues_list = torch.cat(trues_list, dim=0)
    unrel_pseudo_list = torch.cat(unrel_pseudo_list, dim=0)
    compare = (pseudo_labels_list.max(dim=1)[1] == trues_list) * 1
    compare_unrel = (unrel_pseudo_list.max(dim=1)[1] == trues_list) * 1

    all_selected = pseudo_filtered_list.sum().numpy()
    print('Selected: {} ({:.2f}%)'.format(
        all_selected,
        100 * all_selected/len(pseudo_labels_list)))
    print('Per-Class selected: ', [int((trues_list[pseudo_filtered_list==1]==i).sum()) for i in range(args.num_class)])
    print('Max prototype prediction: {}'.format(unrel_pseudo_list.max(dim=1)[0].mean()))
    print('Rho {:.2f}, Pseudo Acc: {:.2f}, Filtered Acc: {:.2f}, Unrel Acc: {:.2f}'.format(
        rho,
        int(compare.sum()) / len(trues_list),
        int((compare * pseudo_filtered_list).sum()) / float(pseudo_filtered_list.sum()),
        int((compare_unrel * (1 - pseudo_filtered_list)).sum()) / float((1 - pseudo_filtered_list).sum())
        ))
    # print('Per Class Accuracy: ', 1)


def warmup(epoch, net, net2, optimizer, dataloader, prototypes):
    net.train()
    net2.train()
    num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1
    for batch_idx, (inputs_w, inputs_s, labels, _) in enumerate(dataloader):
        inputs_w, inputs_s, labels = inputs_w.cuda(), inputs_s.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs_w)
        outputs2 = net2(inputs_w)
        l_ce = CEloss(outputs, labels)
        l_ce2 = CEloss(outputs2, labels)

        loss = l_ce + l_ce2

        if(loss.isnan().any()):
            print("nan in loss")
        penalty = conf_penalty(outputs)
        if(penalty.isnan().any()):
            print("nan in penalty")
        L = loss + penalty
        L.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('%s:%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f  Penalty-loss: %.4f  All-loss: %.4f'
                         % (
                         args.dataset, args.noise_type, epoch, args.num_epochs, batch_idx + 1, num_iter,loss.item(),penalty.item(), L.item()))

def evaluate(loader, model, save = False, best_acc = 0.0):
    model.eval()    # Change model to 'eval' mode.
    
    correct = 0
    total = 0
    for batch_idx, (images, labels) in enumerate(loader):
        images = torch.autograd.Variable(images).cuda()
        logits = model(images)
        outputs = F.softmax(logits, dim=1)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred.cpu() == labels).sum()
    acc = 100*float(correct)/float(total)
    if save:
        if acc > best_acc:
            state = {'state_dict': model.state_dict(),
                     'epoch':epoch,
                     'acc':acc,
            }
            save_path= os.path.join('./', args.noise_type +'best.pth.tar')
            torch.save(state,save_path)
            best_acc = acc
            print(f'model saved to {save_path}!')

    return acc

def test(epoch, net1, net2, prototypes):
    net1.eval()
    net2.eval()
    correct = 0
    correct2 = 0
    correctmean = 0
    correctmax = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)

            score1, predicted = torch.max(outputs1, 1)
            score2, predicted_2 = torch.max(outputs2, 1)
            outputs_mean = (outputs1 + outputs2) / 2
            _, predicted_mean = torch.max(outputs_mean, 1)
            max1_idx = torch.where(score1 > score2)
            predicted_max = predicted_2.clone().detach()
            predicted_max[max1_idx] = predicted[max1_idx]

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
            correct2 += predicted_2.eq(targets).cpu().sum().item()
            correctmean += predicted_mean.eq(targets).cpu().sum().item()
            correctmax += predicted_max.eq(targets).cpu().sum().item()
    acc = 100. * correct / total
    acc2 = 100. * correct2 / total
    accmean = 100. * correctmean / total
    accmax = 100. * correctmax / total
    print("| Test Epoch #%d\t Acc Net1: %.2f%%, Acc Net2: %.2f%% Acc Mean: %.2f%% Acc Max: %.2f%%\n" % (epoch, acc, acc2,accmean,accmax))
    test_log.write('Epoch:%d   Accuracy:%.2f\n' % (epoch, acc))
    test_log.flush()


def eval_train(model, all_loss, rho, num_class):
    model.eval()
    losses = torch.zeros(50000)
    targets_list = torch.zeros(50000)
    pred_list = torch.zeros(50000, num_class)
    num_class = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            num_class = outputs.shape[1]
            loss = CE(outputs, targets)
            targets_cpu = targets.cpu()
            pred = torch.softmax(outputs, dim=1).cpu()
            for b in range(inputs.size(0)):
                losses[index[b]] = loss[b]
                targets_list[index[b]] = targets_cpu[b]
                pred_list[index[b]] = pred[b]

    losses = (losses - losses.min()) / (losses.max() - losses.min())
    all_loss.append(losses)

    input_loss = losses.reshape(-1, 1)

    prob = np.zeros(targets_list.shape[0])
    idx_chosen_sm = []
    min_len = 1e10
    for j in range(num_class):
        indices = np.where(targets_list.cpu().numpy()==j)[0]
        # torch.where will cause device error
        if len(indices) == 0:
            continue
        bs_j = targets_list.shape[0] * (1. / num_class)
        pseudo_loss_vec_j = losses[indices]
        sorted_idx_j = pseudo_loss_vec_j.sort()[1].cpu().numpy()
        partition_j = max(min(int(math.ceil(bs_j*rho)), len(indices)), 1)
        # at least one example
        idx_chosen_sm.append(indices[sorted_idx_j[:partition_j]])
        min_len = min(min_len, partition_j)
    # idx_chosen_sm = [x[:min_len] for x in idx_chosen_sm]
    idx_chosen_sm = np.concatenate(idx_chosen_sm)
    prob[idx_chosen_sm] = 1
    
    # pseudo_labels = F.one_hot(targets_list.long(), num_class)
    # prod = pseudo_labels * pred_list
    # high_conf_cond = (pseudo_labels * pred_list).sum(dim=1) > args.tau
    # prob[high_conf_cond] = 1

    return prob, all_loss


def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u * float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u) ** 2)

        return Lx, Lu, linear_rampup(epoch, warm_up)


class NegEntropy(object):
    def __call__(self, outputs):
        outputs = outputs.clamp(min=1e-12)
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))


def create_model():
    model = DualNet(args)
    # model = SupConWideResNet(num_class=args.num_class)
    model = model.cuda()
    return model


stats_log = open('./checkpoint/%s_%s_%s' % (args.dataset, args.noise_type, args.num_epochs) + '_stats.txt', 'w')
test_log = open('./checkpoint/%s_%s_%s' % (args.dataset, args.noise_type, args.num_epochs) + '_acc.txt', 'w')

warm_up = args.pretrain_ep

loader = dataloader.cifarn_dataloader(args.dataset, noise_type=args.noise_type, noise_path=args.noise_path,
                                      is_human=args.is_human, batch_size=args.batch_size, num_workers=8, \
                                      root_dir=args.data_path, log=stats_log,
                                      noise_file='%s/%s.json' % (args.data_path,args.noise_type))

print('| Building net')
dualnet = create_model()
cudnn.benchmark = True

criterion = SemiLoss()
conf_penalty = NegEntropy()
optimizer1 = optim.SGD([{'params': dualnet.net1.parameters()},
                        {'params': dualnet.net2.parameters()}
                        ], lr=args.lr, momentum=0.9, weight_decay=5e-4)
# optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
# optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

fmix = FMix()

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
CEsoft = CE_Soft_Label()

prototypes = torch.zeros(args.num_class, 128).cuda()
labeled_trainloader = None
unlabeled_trainloader = None

all_loss = [[], []]  # save the history of losses from two networks
# with torch.autograd.detect_anomaly():
best_acc = 0
for epoch in range(args.num_epochs + 1):

    adjust_learning_rate(args, optimizer1, epoch)
    test_loader = loader.run('test')
    eval_loader = loader.run('eval_train')

    if epoch < warm_up:
        warmup_trainloader, noisy_labels = loader.run('warmup')
        if CEsoft.confidence is None:
            CEsoft.init_confidence(noisy_labels, args.num_class)

        print('Warmup Net1')
        warmup(epoch, dualnet.net1, dualnet.net2, optimizer1, warmup_trainloader, prototypes)

    else:
        rho = args.rho_start + (args.rho_end - args.rho_start) * linear_rampup2(epoch, args.warmup_ep)
        prob1, all_loss[0] = eval_train(dualnet.net1, all_loss[0], rho, args.num_class)

        pred1 = (prob1 > args.p_threshold)

        # print('Train Net1')
        labeled_trainloader, noisy_labels = loader.run('train', pred1, prob1)  # co-divide

        train(epoch, dualnet.net1, dualnet.net2, optimizer1, labeled_trainloader, unlabeled_trainloader, prototypes)  # train net1

        # print('\nTrain Net2')
        # labeled_trainloader, unlabeled_trainloader = loader.run('train', pred1, prob1)  # co-divide
        # train(epoch, net2, net1, optimizer2, labeled_trainloader, unlabeled_trainloader, queue)  # train dualnet.net2

    test(epoch, dualnet.net1, dualnet.net2, prototypes)
    torch.save(dualnet, os.path.join('./checkpoint/'+str(args.noise_type)+'_'+str(epoch)+'.pth'))
    best_acc = evaluate(test_loader, dualnet, save = False, best_acc = best_acc)