import torch
import torch.nn as nn
import torchvision


def conv(kernel_size,inchannel, outchannel, stride):
    return nn.Conv1d(inchannel, outchannel, kernel_size=(kernel_size,), stride=(stride,), padding=(kernel_size//2,),bias=False)


class ResBlock(nn.Module):
    def __init__(self,kernel_size, inchannel, outchannel, stride):
        super().__init__()
        self.conv1 =conv(kernel_size,inchannel, outchannel, stride)
        self.bn1 = nn.BatchNorm1d(outchannel)
        self.relu1 = nn.ReLU()
        self.conv2 = conv(kernel_size,outchannel, outchannel, 1)
        self.bn2 = nn.BatchNorm1d(outchannel)
        self.relu2 = nn.ReLU()
        self.downsample = None if inchannel==outchannel else conv(kernel_size,inchannel,outchannel,stride)

    def forward(self,x):
        res = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        if self.downsample is not None:
            res = self.downsample(res)
        return x+res


def ConvBlock(in_channel,out_channel,kernel_size,stride,padding):
    return nn.Sequential(
        nn.Conv1d(in_channel, out_channel, kernel_size=(kernel_size,), stride=(stride,), padding=(padding,),bias=False),
        nn.BatchNorm1d(out_channel),
        nn.ReLU(),
    )


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.mean = 0.8849385890565288
        self.std = 0.2190219816947061
        self.layer1 = ConvBlock(2,4,7,3,0)
        self.layer2 = ConvBlock(4,16,7,3,0)
        self.layer3 = ConvBlock(16,64,5,2,0)
        self.layer4 = ConvBlock(64,128,5,2,0)
        self.pred = nn.Sequential(
            nn.Linear(212,212,bias=False),
            nn.BatchNorm1d(212),
            nn.ReLU(),
            nn.Linear(212,1)
        )

    def forward(self,x):
        B,C,W = x.shape
        x = (x-self.mean)/self.std
        avg = torch.mean(x,dim=0,keepdim=True).expand_as(x)
        x = torch.cat([x,avg],1)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        feat = torch.cat([x1[:,:,0],x2[:,:,0],x3[:,:,0],x4[:,:,0]],dim=1)
        pred = self.pred(feat)
        return pred


if __name__=="__main__":
    net = Net()
    x = torch.randn((8,1,81,))
    y = net(x)
    print(y.shape)

