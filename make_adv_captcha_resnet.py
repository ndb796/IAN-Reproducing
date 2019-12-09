import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from scipy import ndimage
from torchvision.utils import save_image
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import os
import argparse
import hashlib
import time
import numpy as np
import copy
import random

from tqdm import tqdm
from models import *

adv_dataset_train_path = 'adv_dataset/train'
adv_dataset_test_path = 'adv_dataset/test'

if not os.path.exists(adv_dataset_train_path):
    os.makedirs(adv_dataset_train_path)

if not os.path.exists(adv_dataset_test_path):
    os.makedirs(adv_dataset_test_path)

epsilon = 0.0314
p = 0.8
max_iter = 10

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def median_filter(x):
    x = ndimage.median_filter(x.detach().cpu().numpy(), 5)
    x = torch.from_numpy(x).to(device)
    return x

class IANAttack(object):
    def __init__(self, model):
        self.model = model

    def perturb(self, x_natural, y):
        x = x_natural.detach()
        temp_epsilon = epsilon
        count = 0

        temp_set = list(range(len(classes)))
        temp_set.remove(y[0].item())
        deceiving_class = torch.tensor([random.choice(temp_set)]).to(device)
        
        with torch.enable_grad():
          while True:
              outputs = self.model(median_filter(x))
              _, predicted = outputs.max(1)
              correct = predicted.eq(y).sum().item()

              if correct != 1:
                  break

              while True:
                  with torch.enable_grad():
                      x.requires_grad_()
                      outputs = self.model(x)
                      _, predicted = outputs.max(1)
                      correct = predicted.eq(deceiving_class).sum().item()

                      softmax_result = F.softmax(outputs, dim=1)
                      loss = F.cross_entropy(outputs, deceiving_class, size_average=False)
                      confidence = softmax_result[0][deceiving_class].item()
                    
                      grad = torch.autograd.grad(loss, [x])[0]
                      x = x.detach() - temp_epsilon * torch.sign(grad.detach())

                      if correct == 1 and confidence >= p:
                          break

                      count += 1
                      if count >= max_iter:
                          break

              temp_epsilon += 0.01
              if count >= max_iter:
                  break
        return x

def attack(x, y, model, adversary):
    model_copied = copy.deepcopy(model)
    model_copied.eval()
    adversary.model = model_copied
    adv = adversary.perturb(x, y)
    return adv

net = ResNet18()
net = net.to(device)
net = torch.nn.DataParallel(net)
file_name = 'basic_train_resnet'
checkpoint = torch.load('./checkpoint/ckpt.t7' + file_name)
net.load_state_dict(checkpoint['net'])
cudnn.benchmark = True

adversary = IANAttack(net)
criterion = nn.CrossEntropyLoss()

def make_adv_captcha(epoch):
    net.eval()
    benign_correct = 0
    benign_filter_correct = 0
    adv_correct = 0
    adv_filter_correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)

            with torch.no_grad():
                outputs = net(inputs)

                _, predicted = outputs.max(1)
                benign_correct += predicted.eq(targets).sum().item()
                # print('Benign Accuracy: ', str(predicted.eq(targets).sum().item() / targets.size(0)))

                outputs = net(median_filter(inputs))

                _, predicted = outputs.max(1)
                benign_filter_correct += predicted.eq(targets).sum().item()
                # print('Benign Filter Accuracy: ', str(predicted.eq(targets).sum().item() / targets.size(0)))

            with torch.no_grad():
                adv = adversary.perturb(inputs, targets)
                adv_outputs = net(adv)

                _, predicted = adv_outputs.max(1)
                adv_correct += predicted.eq(targets).sum().item()
                # print('Adv Accuracy: ', str(predicted.eq(targets).sum().item() / targets.size(0)))

                adv_outputs = net(median_filter(adv))

                label_string = str(targets.cpu().numpy()[0])
                hash_object = hashlib.md5(label_string.encode() + str(time.time()).encode())
                name = label_string + '_' + hash_object.hexdigest() + '.png'

                if batch_idx < 40000:
                    file_name = adv_dataset_train_path + '/' + name
                    save_image(adv[0], file_name)
                else:
                    file_name = adv_dataset_test_path + '/' + name
                    save_image(adv[0], file_name)     

                _, predicted = adv_outputs.max(1)
                adv_filter_correct += predicted.eq(targets).sum().item()
                # print('Adv Filter Accuracy: ', str(predicted.eq(targets).sum().item() / targets.size(0)))
 
            if batch_idx % 100 == 0:
                print('Index:', batch_idx)

    print('Total Benign Accuarcy:', 100. * benign_correct / total)
    print('Total Benign Filter Accuarcy:', 100. * benign_filter_correct / total)
    print('Total Adv Accuarcy:', 100. * adv_correct / total)
    print('Total Adv Filter Accuracy:', 100. * adv_filter_correct / total)

make_adv_captcha(0)
