from scipy import ndimage
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

from torch.utils.data import DataLoader, Dataset
from PIL import Image

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from tqdm import tqdm
from models import *

learning_rate = 0.1
epsilon = 0.0314
k = 7
alpha = 0.00784
file_name = 'adv_training_4'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

TRAIN_DATASET_PATH = 'adv_dataset_2' + os.path.sep + 'train'
TEST_DATASET_PATH = 'adv_dataset_2' + os.path.sep + 'test'

TRAIN_BATCH_SIZE = 256
TEST_BATCH_SIZE = 128

class MyDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.train_image_file_paths = [os.path.join(folder, image_file) for image_file in os.listdir(folder)]
        self.transform = transform

    def __len__(self):
        return len(self.train_image_file_paths)

    def __getitem__(self, idx):
        image_root = self.train_image_file_paths[idx]
        image_name = image_root.split(os.path.sep)[-1]
        image = Image.open(image_root)
        if self.transform is not None:
            image = self.transform(image)
        target = image_name.split('_')[0]
        label = torch.tensor(int(target))
        return image, label

def get_train_data_loader():
    dataset = MyDataset(TRAIN_DATASET_PATH, transform=transform_train)
    return DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

def get_test_data_loader():
    dataset = MyDataset(TEST_DATASET_PATH, transform=transform_test)
    return DataLoader(dataset, batch_size=TEST_BATCH_SIZE, shuffle=True)

train_loader = get_train_data_loader()
test_loader = get_test_data_loader()

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def median_filter(x):
    x = ndimage.median_filter(x.detach().cpu().numpy(), 2)
    x = torch.from_numpy(x).to(device)
    return x

class LinfPGDAttack(object):
    def __init__(self, model):
        self.model = model

    def perturb(self, x_natural, y):
        x = x_natural.detach()
        x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
        for i in range(k):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = F.cross_entropy(logits, y, size_average=False)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + alpha * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - epsilon), x_natural + epsilon)
            x = torch.clamp(x, 0, 1)
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
cudnn.benchmark = True

adversary = LinfPGDAttack(net)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0002)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        benign_outputs = net(median_filter(inputs))
        loss = criterion(benign_outputs, targets)
        loss.backward()

        optimizer.step()
        train_loss += loss.item()
        _, predicted = benign_outputs.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx % 5 == 0:
            print('Benign Accuracy: ', str(predicted.eq(targets).sum().item() / targets.size(0)))
            print('Loss:', loss.item())

    print('Total Accuarcy:', 100. * correct / total)
    print('Total Loss:', train_loss)

def test(epoch):
    net.eval()
    benign_loss = 0
    adv_loss = 0
    correct = 0
    benign_correct = 0
    adv_correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
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

            if batch_idx % 5 == 0:
                print('Loss:', loss.item())

    print('Total Benign Accuarcy:', 100. * benign_correct / total)
    print('Total Benign Loss:', benign_loss)

    state = {
        'net': net.state_dict()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt.t7' + file_name)
    print('Model Saved')

def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

for epoch in range(0, 200):
    adjust_learning_rate(optimizer, epoch)
    train(epoch)
    test(epoch)
