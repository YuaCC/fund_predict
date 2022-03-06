import torch
from torch.optim import Adam,SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import dataloader
from model.net import Net


def get_kl_loss(x):
    mean = x.mean()
    std = x.std()
    mean2 = mean*mean
    std2 = std*std
    kl_loss = 0.5 * (-torch.log(std2)+mean2+std2)
    return kl_loss


def train(model,train_loader,lr_scheduler):
    model = model.train()
    print(f"epoch {i}")
    for x, y, y0 in train_loader:
        x, y, y0 = x.cuda(), y.cuda(), y0.cuda()
        optim.zero_grad()
        money_in = model(x)
        yinli_loss = -(money_in * (y / y0 - 1)).mean()
        kl_loss = get_kl_loss(money_in)
        loss = yinli_loss + regular_factor * kl_loss
        loss.backward()
        optim.step()
        # print(loss.item())
    lr_scheduler.step()

def val(model,val_loader):
    model = model.eval()
    money_in_sum = 0.0001
    yinli_sum = 0
    results = []
    for x, y, y0 in val_loader:
        x, y, y0 = x.cuda(), y.cuda(), y0.cuda()
        money_in = model(x)
        yinli = money_in * (y / y0 - 1)
        money_in_sum += money_in.abs().sum().item()
        yinli_sum += yinli.sum().item()
        yinlilv = (y / y0 - 1)
        result = torch.cat([money_in, yinlilv], dim=1).detach().cpu().numpy().tolist()
        results.extend(result)

    yinlilv_avg = yinli_sum / money_in_sum
    results.sort(key=lambda x: x[0], reverse=True)
    print(yinlilv_avg)
    print(results[:15])
    print(results[-15:])
    return yinlilv_avg

if __name__ == '__main__':
    dataset_folder = './data/dataset'
    files_list = './data/trainval_files.txt'
    weight_file = 'best_model.pkl'
    model = Net().cuda()
    train_loader,val_loader,test_loader = dataloader(dataset_folder,files_list,64)
    optim = Adam(model.parameters(), lr=0.1)
    # lr_scheduler = MultiStepLR(optim,milestones=[3,6],gamma=0.1)
    lr_scheduler = CosineAnnealingLR(optim, 6)
    epoch = 36
    eps = 0.0001
    regular_factor = 0.001
    best_yinlilv = 0
    for i in range(epoch):
        train(model,train_loader,lr_scheduler)
        yinlilv_avg = val(model,val_loader)
        if yinlilv_avg > best_yinlilv:
            best_yinlilv = yinlilv_avg
            torch.save(model.state_dict(), weight_file)
    print(f"best_yinli={best_yinlilv}")
