
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from model import myAlexNet

from torch import optim

from torch import nn

import torch

import matplotlib.pyplot as plt
#
# import sys
# f = open('alex_train.log', 'a')
# sys.stdout = f



transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((227,227)),transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
train_set = torchvision.datasets.CIFAR10("/Users/aqizhou/PycharmProjects/pytorch_study/dataset",train=True,transform=transform,download=True)
test_set  = torchvision.datasets.CIFAR10("/Users/aqizhou/PycharmProjects/pytorch_study/dataset",train=False,transform=transform,download=True)

train_loader = DataLoader(train_set,batch_size=64)
test_loader = DataLoader(test_set,batch_size=64)



device = torch.device("mps")

#模型定义
net = myAlexNet()
net = net.to(device)

#损失函数定义
lossFunc = nn.CrossEntropyLoss()
lossFunc = lossFunc.to(device)

#优化器
optimizer = optim.Adam(net.parameters(),lr=0.001)

train_loss = []
test_accuracy = []

for epoch in range(2):
    net.train()
    print("-------------第{}轮 训练-------------".format(epoch+1))
    for i,data in enumerate(train_loader):
        imgs,targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        output = net(imgs)
        # targets_pre = torch.argmax(output,dim=1)

        # print("imgs.shape:{},output.shape:{},targets.shape:{}".farmat(imgs.shape,output.shape,targets.shape))
        #
        # print("output:",output)
        # print("___________________")
        # print("targets:",targets)

        loss = lossFunc(output,targets)


        train_loss.append(loss.item())

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if i%100==0:
            print("训练次数：{}，loss:{}".format(i+1,loss.item()))

        # print(train_loss)


    net.eval()
    accuracy = 0
    max_accuracy = 0
    total_correct=0
    for data in test_loader:
        imgs,targets = data

        imgs = imgs.to(device)
        targets = targets.to(device)
        output = net(imgs)

        with torch.no_grad():
            labels_pre = torch.argmax(output,dim=1)
            total_correct += (labels_pre==targets).sum()
    accuracy = total_correct/len(test_set)
    accuracy = accuracy.item()
    test_accuracy.append(accuracy)
    print("test_accuracy:",test_accuracy)
    print("accuracy:{}".format(accuracy))


    #模型保存
    if accuracy > max_accuracy:
        torch.save(net,"model.plt")
        print("模型保存成功！")



x1 = range(0, 10)
y1 = test_accuracy
plt.plot(x1, y1)
plt.title("test accuracy of alexNet")
plt.xlabel("epoches")
plt.ylabel("test accuracy")
plt.savefig('test_accuracy.png')
plt.close()


x2 = range(0,10)
y2 = train_loss
plt.plot(x2,y2)
plt.title("train loss of alexNet")
plt.xlabel("epoches")
plt.ylabel("train loss")
plt.savefig("train_loss.png")
plt.close()









import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from model import myAlexNet

from torch import optim

from torch import nn

import torch

import matplotlib.pyplot as plt
#
# import sys
# f = open('alex_train.log', 'a')
# sys.stdout = f



transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((227,227)),transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
train_set = torchvision.datasets.CIFAR10("/Users/aqizhou/PycharmProjects/pytorch_study/dataset",train=True,transform=transform,download=True)
test_set  = torchvision.datasets.CIFAR10("/Users/aqizhou/PycharmProjects/pytorch_study/dataset",train=False,transform=transform,download=True)

train_loader = DataLoader(train_set,batch_size=64)
test_loader = DataLoader(test_set,batch_size=64)



device = torch.device("cuda")

#模型定义
net = myAlexNet()
net = net.to(device)

#损失函数定义
lossFunc = nn.CrossEntropyLoss()
lossFunc = lossFunc.to(device)

#优化器
optimizer = optim.Adam(net.parameters(),lr=0.001)

train_loss = []
test_accuracy = []

max_accuracy = 0

for epoch in range(20):
    net.train()
    print("-------------第{}轮 训练-------------".format(epoch+1))
    for i,data in enumerate(train_loader):
        imgs,targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        output = net(imgs)
        # targets_pre = torch.argmax(output,dim=1)

        # print("imgs.shape:{},output.shape:{},targets.shape:{}".farmat(imgs.shape,output.shape,targets.shape))
        #
        # print("output:",output)
        # print("___________________")
        # print("targets:",targets)

        loss = lossFunc(output,targets)


        train_loss.append(loss.item())

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if i%100==0:
            print("训练次数：{}，loss:{}".format(i+1,loss.item()))

        # print(train_loss)


    net.eval()
    accuracy = 0
    total_correct=0
    for data in test_loader:
        imgs,targets = data

        imgs = imgs.to(device)
        targets = targets.to(device)
        output = net(imgs)

        with torch.no_grad():
            labels_pre = torch.argmax(output,dim=1)
            total_correct += (labels_pre==targets).sum()
    accuracy = total_correct/len(test_set)
    accuracy = accuracy.item()
    test_accuracy.append(accuracy)
    print("test_accuracy:",test_accuracy)
    print("accuracy:{}".format(accuracy))


    #模型保存
    if accuracy > max_accuracy:
        max_accuracy = accuracy
        torch.save(net,"model.plt")
        print("模型保存成功！")



x1 = range(0, 20)
y1 = test_accuracy
plt.plot(x1, y1)
plt.title("test accuracy of alexNet")
plt.xlabel("epoches")
plt.ylabel("test accuracy")
plt.savefig('test_accuracy.png')
plt.close()


x2 = range(0,20)
y2 = train_loss
plt.plot(x2,y2)
plt.title("train loss of alexNet")
plt.xlabel("epoches")
plt.ylabel("train loss")
plt.savefig("train_loss.png")
plt.close()


























