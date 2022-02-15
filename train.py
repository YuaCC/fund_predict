import torch
from dataset import dataloader
from model.net import Net
from torchvision.transforms import ToTensor
from torch.optim import Adam,SGD
import numpy as np
if __name__=='__main__':
    model = Net().cuda()
    train_loader = dataloader('./data/dataset','./data/files1.txt',None,True,batch_size=64)
    val_loader = dataloader('./data/dataset','./data/files2.txt',None,True,batch_size=64)
    optim = Adam(model.parameters(),lr=10000)
    epoch = 30
    for i in range(epoch):
        model = model.train()
        for x,y,y0 in train_loader:
            x,y,y0 = x.cuda(),y.cuda(),y0.cuda()
            optim.zero_grad()
            pred = model(x)
            loss = -(pred * y/y0).sum()/(pred.abs().sum()+1)
            loss.backward()
            optim.step()
            print(loss.item())

        model=model.eval()
        money_in_sum = np.zeros((3,),dtype=np.float)
        money_out_sum = np.zeros((3,),dtype=np.float)

        for x,y,y0 in val_loader:
            x,y,y0 = x.cuda(),y.cuda(),y0.cuda()
            money_in = model(x)
            money_out = money_in/y0*y
            mask = money_in<=0
            money_in[mask] = 0
            money_out[mask] =0

            money_in_sum += money_in.detach().cpu().sum(dim=0).numpy()
            money_out_sum += money_out.detach().cpu().sum(dim=0).numpy()
            print(money_out_sum/money_in_sum)



