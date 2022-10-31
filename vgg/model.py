from torch import nn
import torch


class myVGG(nn.Module):
    def __init__(self,features,num_classes=1000):
        super(myVGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(7*7*512,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self,input):
        out = self.features(input)
        flatten = nn.Flatten()
        out = flatten(out)
        out = self.classifier(out)
        return out




cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def create_features(cfg):
    features = []
    channels_in = 3
    for i in cfg:
        if i=='M':
            features += [nn.MaxPool2d(kernel_size=2,stride=2)]
        else:
            features += [nn.Conv2d(in_channels=channels_in,out_channels=i,kernel_size=3,padding=1),nn.ReLU()]
            channels_in = i

    return nn.Sequential(*features)





if __name__ == '__main__':
    cfg = cfgs["vgg16"]
    features = create_features(cfg)
    vgg =myVGG(features,num_classes=10)

    input = torch.ones((1,3,224,224))
    print(input)
    output = vgg(input)
    print(output)



