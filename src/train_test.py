import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 1/3 and 2/3 epochs"""
    if (epoch*3==args.epochs) or (epoch*3==2*args.epochs):
        lr = args.lr * (0.1 ** (epoch*3//args.epochs))
        print("Changing Learning Rate to {}".format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def adjust_learning_rate_iccv(args,optimizer, epoch):
    lr = args.lr
    if (epoch*2==args.epochs) or (epoch*4==3*args.epochs):
        lr = args.lr * 0.1
        print("Changing Learning Rate to {}".format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return lr

def adjust_learning_rate_nips(args,optimizer, epoch):
    lr = args.lr
    if (epoch==160) or (epoch==240):
        lr = args.lr*0.1
        print("Changing Learning Rate to {}".format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return lr


def train(args,model,device,train_loader,optimizer,epoch,classes=None):
    model.train()
    train_loss = 0
    correct = 0
    if classes is not None:
        class_correct = np.zeros(len(classes))
        class_total = np.zeros(len(classes))

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        if device == torch.device('cuda'):
            data = data.type(torch.cuda.FloatTensor)
            target = target.type(torch.cuda.LongTensor)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        train_loss += F.nll_loss(output, target,reduction='sum').item()
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        
        if classes is not None:
            c = pred.eq(target.view_as(pred))
            for i,label in enumerate(target):
                class_correct[label] += c[i]
                class_total[label] += 1
        else:
            correct += pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


    train_loss /= len(train_loader.dataset)
    if classes is not None:
        train_acc = 100.*sum(class_correct)/sum(class_total)
    else:
        train_acc = 100.*correct/len(train_loader.dataset)   

    return train_loss,train_acc            
                
            
def test(args,model,device,test_loader,classes=None, verbose=True):
    model.eval()
    test_loss = 0
    correct = 0
    if classes is not None:
        class_correct = np.zeros(len(classes))
        class_total = np.zeros(len(classes))

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if device == torch.device('cuda'):
                data = data.type(torch.cuda.FloatTensor)
                target = target.type(torch.cuda.LongTensor)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            
            if classes is not None:
                c = pred.eq(target.view_as(pred))
                for i,label in enumerate(target):
                    class_correct[label] += c[i]
                    class_total[label] += 1
            else:
                correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    if classes is not None:
        test_acc =  100.*sum(class_correct)/sum(class_total)
        for i in range(len(classes)):
            print('Accuracy of %s : %2d%% out of %d cases' % (classes[i], 100 * class_correct[i] / class_total[i], class_total[i]))
        print('Total Accuracy:',test_acc)
    else:
        test_acc = 100.*correct/len(test_loader.dataset)
        if verbose: print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),test_acc))     
    
    return test_loss,test_acc 



def train_sleepnet(model,trainloader_v1,optimizer,criterion,classes,mode='gpu'):
    
    # set the model as train mode
    model = model.train()
    
    train_loss = 0.0
    train_counter = 0
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    for i, data in enumerate(trainloader_v1, 0):
        
        # get the inputs
        inputs, targets = data

        # wrap them in Variable
        inputs, targets = Variable(inputs), Variable(targets)
        if mode == 'gpu':
            inputs, targets = inputs.cuda(), targets.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == targets).squeeze().cpu().data.numpy()
        for i, label in enumerate(targets):
            class_correct[label] += c[i]
            class_total[label] += 1
        
        train_loss += (loss.data[0] * inputs.size(0))
        train_counter += inputs.size(0)
    
    train_acc = sum(class_correct)/sum(class_total)
    return train_acc, train_loss/train_counter


def test_sleepnet(model, testloader_v1, classes, criterion, mode='gpu'):
    
    # set model in eval mode
    model = model.eval()
    
    test_loss = 0.0
    test_counter = 0
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    for data in testloader_v1:
        # get the inputs
        inputs, targets = data

        # wrap them in Variable
        inputs = Variable(inputs, volatile=True)
        if mode == 'gpu':
            inputs, targets = inputs.cuda(), targets.cuda()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += (loss.data[0] * inputs.size(0))
        test_counter += inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == targets).squeeze().cpu().data.numpy()
        for i, label in enumerate(targets):
            class_correct[label] += c[i]
            class_total[label] += 1
        
        
                
    for i in range(len(classes)):
        print('Accuracy of %s : %2d%% out of %d cases' % (classes[i], 100 * class_correct[i] / class_total[i], class_total[i]))
    print('Total Accuracy:', sum(class_correct)/sum(class_total))
    test_acc = sum(class_correct)/sum(class_total)
    return test_acc, test_loss/test_counter


