import os
import torch
import copy
import random
import json
from data.utils import download_url, check_integrity
from utils.randaug import *
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict


class cifarn_dataset(Dataset):
    def __init__(self,  dataset,  noise_type, noise_path, root_dir, transform, mode, transform_s=None, is_human=True, noise_file='',
                 pred=[], probability=[],probability2=[] ,log='', print_show=False, r =0.2 , noise_mode = 'cifarn'):
        self.dataset = dataset
        self.transform = transform
        self.transform_s = transform_s
        self.mode = mode
        self.noise_type = noise_type
        self.noise_path = noise_path
        idx_each_class_noisy = [[] for i in range(10)]
        self.print_show = print_show
        self.noise_mode = noise_mode
        self.r = r
        self.transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8} # class transition for asymmetric noise

        if dataset == 'cifar10':
            self.nb_classes = 10
            idx_each_class_noisy = [[] for i in range(10)]
        elif dataset == 'cifar100':
            self.nb_classes = 100
            idx_each_class_noisy = [[] for i in range(100)]
        if self.mode == 'test':
            if dataset == 'cifar10':
                test_dic = unpickle('%s/test_batch' % root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic['labels']
            elif dataset == 'cifar100':
                test_dic = unpickle('%s/test' % root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic['fine_labels']
        else:
            train_data = []
            train_label = []
            if dataset == 'cifar10':
                for n in range(1, 6):
                    dpath = '%s/data_batch_%d' % (root_dir, n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label + data_dic['labels']
                train_data = np.concatenate(train_data)
            elif dataset == 'cifar100':
                train_dic = unpickle('%s/train' % root_dir)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))
            self.train_labels = train_label

            # if noise_type is not None:
            if os.path.exists(noise_file):
                noise_label = json.load(open(noise_file,"r"))
                self.train_noisy_labels = noise_label
                self.noise_or_not = np.transpose(self.train_noisy_labels) != np.transpose(self.train_labels)
            else:    #inject noise   
                if self.noise_mode=='sym' or self.noise_mode =='asym':
                    noise_label = []
                    idx = list(range(50000))
                    random.shuffle(idx)
                    num_noise = int(self.r*50000)            
                    noise_idx = idx[:num_noise]
                    for i in range(50000):
                        if i in noise_idx:
                            if self.noise_mode=='sym':
                                if dataset=='cifar10': 
                                    noiselabel = random.randint(0,9)
                                elif dataset=='cifar100':    
                                    noiselabel = random.randint(0,99)
                                noise_label.append(noiselabel)
                            elif self.noise_mode=='asym':   
                                noiselabel = self.transition[train_label[i]]
                                noise_label.append(noiselabel)                    
                        else:    
                            noise_label.append(train_label[i])   
                    self.train_noisy_labels = noise_label
                    self.noise_or_not = np.transpose(self.train_noisy_labels) != np.transpose(self.train_labels)
                    print("save noisy labels to %s ..."%noise_file)        
                    json.dump(noise_label,open(noise_file,"w"))

                elif self.noise_mode == 'cifarn':
                    if noise_type != 'clean':
                        # Load human noisy labels
                        train_noisy_labels = self.load_label()
                        self.train_noisy_labels = train_noisy_labels.tolist()
                        self.print_wrapper(f'noisy labels loaded from {self.noise_path}')
                    
                        for i in range(len(self.train_noisy_labels)):
                            idx_each_class_noisy[self.train_noisy_labels[i]].append(i)
                        class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(10)]
                        self.noise_prior = np.array(class_size_noisy) / sum(class_size_noisy)
                        self.print_wrapper(f'The noisy data ratio in each class is {self.noise_prior}')
                        self.noise_or_not = np.transpose(self.train_noisy_labels) != np.transpose(self.train_labels)
                        self.actual_noise_rate = np.sum(self.noise_or_not) / 50000
                        self.print_wrapper('over all noise rate is ', self.actual_noise_rate)
                    noise_label = train_noisy_labels
                

            if self.mode == 'all_lab':
                self.probability = probability
                self.probability2 = probability2
                self.train_data = train_data
                self.noise_label = noise_label
            elif self.mode == 'all':
                self.train_data = train_data
                self.noise_label = noise_label
            else:
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]

                    clean = (np.array(noise_label) == np.array(train_label))
                    log.write('Numer of labeled samples:%d   AUC (not computed):%.3f\n' % (pred.sum(), 0))
                    log.flush()

                elif self.mode == "unlabeled":
                    pred_idx = (1 - pred).nonzero()[0]

                self.train_data = train_data[pred_idx]
                self.noise_label = [noise_label[i] for i in pred_idx]
                self.print_wrapper("%s data has a size of %d" % (self.mode, len(self.noise_label)))
        self.print_show = False
    
    def print_wrapper(self, *args, **kwargs):
        if self.print_show:
            print(*args, **kwargs)

    def load_label(self):
        # NOTE only load manual training label
        noise_label = torch.load(self.noise_path)
        if isinstance(noise_label, dict):
            if "clean_label" in noise_label.keys():
                clean_label = torch.tensor(noise_label['clean_label'])
                assert torch.sum(torch.tensor(self.train_labels) - clean_label) == 0
                self.print_wrapper(f'Loaded {self.noise_type} from {self.noise_path}.')
                self.print_wrapper(f'The overall noise rate is {1 - np.mean(clean_label.numpy() == noise_label[self.noise_type])}')
            return noise_label[self.noise_type].reshape(-1)
        else:
            raise Exception('Input Error')

    def __getitem__(self, index):
        if self.mode == 'labeled':
            img, target, prob = self.train_data[index], self.noise_label[index], self.probability[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            img2 = self.transform_s(img)
            return img1, img2, target, prob
        elif self.mode == 'unlabeled':
            img = self.train_data[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            img2 = self.transform_s(img)
            return img1, img2
        elif self.mode == 'all_lab':
            img, target, prob, prob2 = self.train_data[index], self.noise_label[index], self.probability[index],self.probability2[index]
            true_labels = self.train_labels[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            img2 = self.transform_s(img)
            return img1, img2, target, prob,prob2,true_labels, index
        elif self.mode == 'all':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            if self.transform_s is not None:
                img1 = self.transform(img)
                img2 = self.transform_s(img)
                return img1, img2, target, index
            else:
                img = self.transform(img)
                return img, target, index
        elif self.mode == 'all2':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            img2 = self.transform_s(img)
            return img1, img2, target, index
        elif self.mode == 'test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target

    def __len__(self):
        if self.mode != 'test':
            return len(self.train_data)
        else:
            return len(self.test_data)


class cifarn_dataloader():
    def __init__(self, dataset, noise_type, noise_path, is_human, batch_size, num_workers, root_dir, log,
                 noise_file='',noise_mode='cifarn', r=0.2):
        self.r = r
        self.noise_mode = noise_mode
        self.dataset = dataset
        self.noise_type = noise_type
        self.noise_path = noise_path
        self.is_human = is_human
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        self.noise_file = noise_file
        if self.dataset == 'cifar10':
            self.transform_train = transforms.Compose([
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.ToTensor(), 
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
            self.transform_train_s = copy.deepcopy(self.transform_train)
            self.transform_train_s.transforms.insert(0, RandomAugment(3,5))
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ])
        elif self.dataset == 'cifar100':
            self.transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
            ])
            self.transform_train_s = copy.deepcopy(self.transform_train)
            self.transform_train_s.transforms.insert(0, RandomAugment(3,5))
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
            ])
        self.print_show = True

    def run(self, mode, pred=[], prob=[],prob2=[]):
        if mode == 'warmup':
            all_dataset = cifarn_dataset(dataset=self.dataset, noise_type=self.noise_type, noise_path=self.noise_path,
                                         is_human=self.is_human, root_dir=self.root_dir, transform=self.transform_train,
                                         transform_s=self.transform_train_s, mode="all",
                                         noise_file=self.noise_file, print_show=self.print_show, r=self.r,noise_mode=self.noise_mode)
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)
            self.print_show = False
            # never show noisy rate again
            return trainloader, all_dataset.train_noisy_labels

        elif mode == 'train':
            labeled_dataset = cifarn_dataset(dataset=self.dataset, noise_type=self.noise_type,
                                             noise_path=self.noise_path, is_human=self.is_human,
                                             root_dir=self.root_dir, transform=self.transform_train, mode="all_lab",
                                             noise_file=self.noise_file, pred=pred, probability=prob,probability2=prob2, log=self.log,
                                             transform_s=self.transform_train_s, r=self.r,noise_mode=self.noise_mode)
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True)

            return labeled_trainloader, labeled_dataset.train_noisy_labels

        elif mode == 'test':
            test_dataset = cifarn_dataset(dataset=self.dataset, noise_type=self.noise_type, noise_path=self.noise_path,
                                          is_human=self.is_human,
                                          root_dir=self.root_dir, transform=self.transform_test, mode='test', r=self.r,noise_mode=self.noise_mode)
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return test_loader

        elif mode == 'eval_train':
            eval_dataset = cifarn_dataset(dataset=self.dataset, noise_type=self.noise_type, noise_path=self.noise_path,
                                          is_human=self.is_human,
                                          root_dir=self.root_dir, transform=self.transform_test, mode='all',
                                          noise_file=self.noise_file, r=self.r,noise_mode=self.noise_mode)
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return eval_loader, eval_dataset.noise_or_not
        # never print again
