
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from model import myLenet5

from torch import optim

from torch import nn

import torch

transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((32,32)),transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
train_set = torchvision.datasets.CIFAR10("/Users/aqizhou/PycharmProjects/pytorch_study/dataset",train=True,transform=transform,download=True)
test_set  = torchvision.datasets.CIFAR10("/Users/aqizhou/PycharmProjects/pytorch_study/dataset",train=False,transform=transform,download=True)

train_loader = DataLoader(train_set,batch_size=64)
test_loader = DataLoader(test_set,batch_size=64)

device = torch.device("mps")

#模型定义
net = myLenet5()
net = net.to(device)

#损失函数定义
lossFunc = nn.CrossEntropyLoss()
lossFunc = lossFunc.to(device)

#优化器
optimizer = optim.SGD(net.parameters(),lr=0.01)


for epoch in range(50):
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

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if i%100==0:
            print("训练次数：{}，loss:{}".format(i+1,loss))

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
    print("accuracy:{}".format(accuracy))

#模型保存
torch.save(net,"model.plt")
print("模型保存成功！")
















