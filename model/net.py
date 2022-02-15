import torch
import torch.nn as nn
import torchvision


def conv7(inchannel, outchannel, stride):
    return nn.Conv1d(inchannel, outchannel, kernel_size=(7,), stride=(stride,), padding=(3,),bias=False)


class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride):
        super().__init__()
        self.conv1 =conv7(inchannel, outchannel, stride)
        self.bn1 = nn.BatchNorm1d(outchannel)
        self.relu1 = nn.ReLU()
        self.conv2 = conv7(outchannel, outchannel, 1)
        self.bn2 = nn.BatchNorm1d(outchannel)
        self.relu2 = nn.ReLU()
        self.downsample = None if inchannel==outchannel else conv7(inchannel,outchannel,stride)

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


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = ResBlock(1,3,3)
        self.layer2 = ResBlock(3,9,3)
        self.layer3 = ResBlock(9,27,3)
        self.layer4 = ResBlock(27,81,3)
        self.pred = nn.Sequential(
            nn.Linear(120,120,bias=False),
            nn.BatchNorm1d(120),
            nn.ReLU(),
            nn.Linear(120,3)
        )


    def forward(self,x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        feat = torch.cat([x1[:,:,-1],x2[:,:,-1],x3[:,:,-1],x4[:,:,-1]],dim=1)
        pred = self.pred(feat)
        return pred

if __name__=="__main__":
    net = Net()
    x = torch.randn((8,1,81,))
    y = net(x)
    print(y.shape)

