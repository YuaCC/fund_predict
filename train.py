import torch
from dataset import dataloader
from model.net import Net
from torchvision.transforms import ToTensor
from torch.optim import Adam,SGD
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR,CosineAnnealingLR
if __name__=='__main__':
    args = [['files1.txt','files2.txt','best_model1.pkl'],['files2.txt','files1.txt','best_model2.pkl']]
    for filea,fileb,modelname in args:
        model = Net().cuda()
        train_loader = dataloader('./data/dataset',f'./data/{filea}',None,True,batch_size=64)
        val_loader = dataloader('./data/dataset',f'./data/{fileb}',None,True,batch_size=64)
        optim = Adam(model.parameters(),lr=0.1)
        # lr_scheduler = MultiStepLR(optim,milestones=[3,6],gamma=0.1)
        lr_scheduler = CosineAnnealingLR(optim,4,0.0001)
        epoch = 12
        best_yinli =0
        for i in range(epoch):
            model = model.train()
            print(f"epoch {i}")
            for x,y,y0 in train_loader:
                x,y,y0 = x.cuda(),y.cuda(),y0.cuda()
                optim.zero_grad()
                pred = model(x)
                loss = -(pred * y/y0).sum()/(pred.sum()+1)
                loss.backward()
                optim.step()
            lr_scheduler.step()

            model=model.eval()
            money_in_sum = 0
            money_out_sum = 0

            for x,y,y0 in val_loader:
                x,y,y0 = x.cuda(),y.cuda(),y0.cuda()
                money_in = model(x)
                money_out = money_in/y0*y

                money_in_sum += money_in.detach().cpu().sum(dim=0).numpy()
                money_out_sum += money_out.detach().cpu().sum(dim=0).numpy()
            print(money_out_sum/money_in_sum)
            if money_in_sum !=0 and money_out_sum/money_in_sum>best_yinli:
                best_yinli = money_out_sum/money_in_sum
                torch.save(model.state_dict(),modelname)
        print(f"best_yinli={best_yinli}")




