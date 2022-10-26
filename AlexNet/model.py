import torch
from torch import nn



class myAlexNet(nn.Module):
    def __init__(self,class_nums=1000):
        super(myAlexNet, self).__init__() #为了方便训练，channel缩小1/2
        self.feature = nn.Sequential(                                        #3*227*227    input
            nn.Conv2d(in_channels=3,out_channels=48,kernel_size=11,stride=4),  #(227-11)/4+1=55 96*55*55
            nn.ReLU(inplace=True), #inplace=True 原地操作
            nn.MaxPool2d(kernel_size=3,stride=2), #(55-3)/2+1=27 96*27*27
            nn.Conv2d(in_channels=48,out_channels=128,kernel_size=5,stride=1,padding=2), #(27-5+2*2)/1+1=27 256*27*27
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),#(27-3)/2+1=13 256*13*13
            nn.Conv2d(in_channels=128,out_channels=192,kernel_size=3,stride=1,padding=1), #(13-3+1*2)/1+1=13 384*13*13
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192,out_channels=192,kernel_size=3,stride=1,padding=1), #(13-3+1*2)/1+1=13 284*13*13
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192,out_channels=128,kernel_size=3,stride=1,padding=1), #(13-3+1*2)/1+1=13 256*13*13
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2), #(13-3)/2+1=6 256*6*6

            nn.Flatten()


        )
        self.classify = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128*6*6,2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048,2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048,class_nums)
        )

    def forward(self,input):
        out = self.feature(input)
        out = self.classify(out)
        return out


if __name__ == '__main__':
    input = torch.ones((1,3,227,227))
    net = myAlexNet(class_nums=10)
    out = net(input)
    print(out)




