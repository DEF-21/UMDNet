import os
from baseline import Mnet
import torch
import random
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from lib.dataset import Data
from lib.data_prefetcher import DataPrefetcher
from torch.nn import functional as F


import pytorch_iou

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
IOU = pytorch_iou.IOU(size_average=True)

def loss_1(score1, score2, score3, score4, s1, s2, s3, s4, label):

    sal_loss1 = F.binary_cross_entropy_with_logits(score1, label, reduction='mean')
    sal_loss2 = F.binary_cross_entropy_with_logits(score2, label, reduction='mean')
    sal_loss3 = F.binary_cross_entropy_with_logits(score3, label, reduction='mean')
    sal_loss4 = F.binary_cross_entropy_with_logits(score4, label, reduction='mean')

    loss1 = sal_loss1 + IOU(s1, label)
    loss2 = sal_loss2 + IOU(s2, label)
    loss3 = sal_loss3 + IOU(s3, label)
    loss4 = sal_loss4 + IOU(s4, label)

    return loss1+loss2+loss3+loss4


if __name__ == '__main__':
    random.seed(118)
    np.random.seed(118)
    torch.manual_seed(118)
    torch.cuda.manual_seed(118)
    torch.cuda.manual_seed_all(118)
    # dataset
    img_root = ''
    save_path = ''
    if not os.path.exists(save_path): os.mkdir(save_path)
    lr = 0.001
    num_workers = 4
    batch_size = 6
    epoch = 100
    lr_dec = [36, 37]
    data = Data(img_root)

    loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    net = Mnet().cuda()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=0.0005, momentum=0.9)


    iter_num = len(loader)
    net.train()
    for epochi in range(1, epoch + 1):
        if epochi in lr_dec:
            lr = lr / 10
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=0.0005,
                                  momentum=0.9)
            print(lr)
        # loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=1)
        # iter_num = len(loader)
        prefetcher = DataPrefetcher(loader)
        rgb, label = prefetcher.next()
        r_sal_loss = 0
        net.zero_grad()
        i = 0

        for j in range(iter_num - 2):
            # while rgb is not None:
            i += 1
            H = rgb.shape[2]
            W = rgb.shape[3]

            score1, score2, score3, score4, s1, s2, s3, s4 = net(rgb)
            # print(score1.shape)
            # print(label.shape)
            sal_loss = loss_1(score1, score2, score3, score4, s1, s2, s3, s4, label)
            r_sal_loss += sal_loss.data
            sal_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i % 5 == 0:
                print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  loss : %5.4f  ||  lr:%6.5f' % (
                    epochi, epoch, i, iter_num, r_sal_loss / 50, lr))
                r_sal_loss = 0
            rgb, label = prefetcher.next()
        if epochi >= 0 and epochi % 2 == 0:
            torch.save(net.state_dict(), '%s/epoch_%dBASELINE.pth' % (save_path, epochi))
    torch.save(net.state_dict(), '%s/baseline.pth' % (save_path))
