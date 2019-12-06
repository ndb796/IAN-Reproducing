import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

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
file_name = 'basic_train_resnet'

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
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

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
    iterator = tqdm(train_loader, ncols=0, leave=False)
    for batch_idx, (inputs, targets) in enumerate(iterator):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        benign_outputs = net(inputs)
        loss = criterion(benign_outputs, targets)
        loss.backward()

        optimizer.step()
        train_loss += loss.item()
        _, predicted = benign_outputs.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx % 5 == 0:
            print('\nBenign Accuracy: ', str(predicted.eq(targets).sum().item() / targets.size(0)))
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
        iterator = tqdm(test_loader, ncols=0, leave=False)
        for batch_idx, (inputs, targets) in enumerate(iterator):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)

            with torch.no_grad():
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                benign_loss += loss.item()

                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                benign_correct += predicted.eq(targets).sum().item()
                print('\nBenign Accuracy: ', str(predicted.eq(targets).sum().item() / targets.size(0)))

            with torch.no_grad():
                adv = adversary.perturb(inputs, targets)
                adv_outputs = net(adv)
                loss = criterion(adv_outputs, targets)
                adv_loss += loss.item()

                _, predicted = adv_outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                adv_correct += predicted.eq(targets).sum().item()
                print('Adv Accuracy: ', str(predicted.eq(targets).sum().item() / targets.size(0)))
            
            if batch_idx % 5 == 0:
                print('Loss:', loss.item())

    print('Total Benign Accuarcy:', 100. * benign_correct / total)
    print('Total Adv Accuarcy:', 100. * adv_correct / total)
    print('Total Accuarcy:', 100. * correct / (total * 2))
    print('Total Benign Loss:', benign_loss)
    print('Total Adv Loss:', adv_loss)

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
