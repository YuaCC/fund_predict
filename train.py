import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import dataloader
from model.net import Net

if __name__ == '__main__':
    args = [['files1.txt', 'files2.txt', 'best_model1.pkl'], ['files2.txt', 'files1.txt', 'best_model2.pkl']]
    for filea, fileb, modelname in args:
        model = Net().cuda()
        train_loader = dataloader('./data/dataset', f'./data/{filea}', "train", batch_size=64)
        val_loader = dataloader('./data/dataset', f'./data/{fileb}', "val", batch_size=64)
        optim = Adam(model.parameters(), lr=0.1)
        # lr_scheduler = MultiStepLR(optim,milestones=[3,6],gamma=0.1)
        lr_scheduler = CosineAnnealingLR(optim, 6, 0.0001)
        epoch = 36
        eps = 0.0001
        kl_factor = 0.001
        best_yinlilv = 0
        for i in range(epoch):
            model = model.train()
            print(f"epoch {i}")
            for x, y, y0 in train_loader:
                x, y, y0 = x.cuda(), y.cuda(), y0.cuda()
                optim.zero_grad()
                money_in = model(x)
                yinli_loss = -(money_in * (y / y0 - 1)).mean()
                mean = money_in.mean()
                std = money_in.std()
                mean_2 = mean*mean
                std_2 = std*std
                kl_loss = 0.5*(-torch.log(std_2)+mean_2+std_2-1).mean()
                loss = yinli_loss + kl_factor*kl_loss
                loss.backward()
                optim.step()
                # print(loss.item())
            lr_scheduler.step()

            model = model.eval()
            money_in_sum = 0.0001
            yinli_sum = 0

            for x, y, y0 in val_loader:
                x, y, y0 = x.cuda(), y.cuda(), y0.cuda()
                money_in = model(x)
                yinli = money_in * (y / y0 - 1)
                money_in_sum += money_in.abs().sum().item()
                yinli_sum += yinli.sum().item()
            yinlilv_avg = yinli_sum / money_in_sum
            print(yinlilv_avg)
            if yinlilv_avg > best_yinlilv:
                best_yinlilv = yinlilv_avg
                torch.save(model.state_dict(), modelname)
        print(f"best_yinli={best_yinlilv}")
