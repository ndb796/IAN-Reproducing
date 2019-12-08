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
k = 7
alpha = 0.00784
p = 0.8

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
              temp_epsilon += 0.01
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
cudnn.benchmark = True

adversary = IANAttack(net)
criterion = nn.CrossEntropyLoss()

def make_adv_captcha(epoch):
    net.eval()
    benign_loss = 0
    adv_loss = 0
    correct = 0
    benign_correct = 0
    adv_correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)

            with torch.no_grad():
                outputs = net(median_filter(inputs))
                loss = criterion(outputs, targets)
                benign_loss += loss.item()

                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                benign_correct += predicted.eq(targets).sum().item()
                print('Benign Accuracy: ', str(predicted.eq(targets).sum().item() / targets.size(0)))

            with torch.no_grad():
                adv = adversary.perturb(inputs, targets)
                adv_outputs = net(median_filter(adv))
                loss = criterion(adv_outputs, targets)
                adv_loss += loss.item()

                label_string = str(targets.cpu().numpy()[0])
                hash_object = hashlib.md5(label_string.encode() + str(time.time()).encode())
                name = label_string + '_' + hash_object.hexdigest() + '.png'

                if batch_idx < 30000:
                    file_name = adv_dataset_train_path + '/' + name
                    save_image(adv[0], file_name)
                else:
                    file_name = adv_dataset_test_path + '/' + name
                    save_image(adv[0], file_name)     

                _, predicted = adv_outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                adv_correct += predicted.eq(targets).sum().item()
                print('Adv Accuracy: ', str(predicted.eq(targets).sum().item() / targets.size(0)))
            
            if batch_idx % 10 == 0:
                print('Index:', batch_idx, 'Loss:', loss.item())

    print('Total Benign Accuarcy:', 100. * benign_correct / total)
    print('Total Adv Accuarcy:', 100. * adv_correct / total)
    print('Total Accuarcy:', 100. * correct / (total * 2))
    print('Total Benign Loss:', benign_loss)
    print('Total Adv Loss:', adv_loss)

make_adv_captcha(0)
