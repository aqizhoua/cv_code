import torch
from torch import nn


class myLenet5(nn.Module):
    def __init__(self):
        super(myLenet5, self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5), #灰度图输入 后改成彩色
            nn.ReLU(), #Sigmoid效果不太好？
            nn.AvgPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2,stride=2),
            nn.Flatten(), #400维
            nn.Linear(16*5*5,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10)
            #最后高斯全连接没写

        )

    def forward(self,input):
        return self.model(input)

if __name__ == '__main__':
    a = torch.ones((1,1,32,32))
    print(a.shape)

    net = myLenet5()
    output = net(a)

    print(output)
