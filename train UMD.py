import os
from UMDNet import Mnet
import torch
import random
import numpy as np

import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from lib.dataset import Data
from lib.data_prefetcher import DataPrefetcher
from torch.nn import functional as F

import pytorch_iou

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
IOU = pytorch_iou.IOU(size_average=True)
mse_loss = torch.nn.MSELoss(size_average=True, reduce=True)

def loss_1(score1, score2, score3, score4, s1, s2, s3, s4, label):

    sal_loss1 = F.binary_cross_entropy_with_logits(score1, label, reduction='mean')
    sal_loss2 = F.binary_cross_entropy_with_logits(score2, label, reduction='mean')
    sal_loss3 = F.binary_cross_entropy_with_logits(score3, label, reduction='mean')
    sal_loss4 = F.binary_cross_entropy_with_logits(score4, label, reduction='mean')

    loss1 = sal_loss1 + IOU(s1, label)
    loss2 = sal_loss2 + IOU(s2, label)
    loss3 = sal_loss3 + IOU(s3, label)
    loss4 = sal_loss4 + IOU(s4, label)

    return loss1 + loss2 + loss3 + loss4


if __name__ == '__main__':
    random.seed(118)
    np.random.seed(118)
    torch.manual_seed(118)
    torch.cuda.manual_seed(118)
    torch.cuda.manual_seed_all(118)

    save_path = ''
    if not os.path.exists(save_path): os.mkdir(save_path)
    lr = 0.001

    num_workers = 4
    batch_size = 6
    epoch = 100
    lr_dec = [80, 81]
    num_params = 0

    # dataset
    # img_root = '/home/ms/Documents/dataset/SOD/RGB/DUTS/DUTS-TE/'
    img_root = ''
    target_root = ''
    data = Data(img_root)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    target_data = Data(target_root)
    source_len = len(data)
    target_len = len(target_data)
    new_target_data = target_data
    for k in range(source_len // target_len + 1):
        new_target_data = ConcatDataset([new_target_data, target_data])
    target_data_final = new_target_data
    target_loader = DataLoader(target_data_final, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    iter_num = len(loader)
    print(len(loader))
    print(len(target_loader))


    # init net
    net = Mnet().cuda()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=0.0005, momentum=0.9)
    net.train()

    for epochi in range(1, epoch + 1):
        if epochi in lr_dec:
            lr = lr / 10
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=0.0005, momentum=0.9)
            print(lr)

        loss_sod_value = 0
        loss_sodt_value = 0


        prefetcher = DataPrefetcher(loader)
        rgb, label = prefetcher.next()

        target_prefetcher = DataPrefetcher(target_loader)
        target_rgb, target_label = target_prefetcher.next()


        net.zero_grad()

        i = 0
        for j in range(iter_num-2):
        # while rgb is not None:
            i += 1
            H = rgb.shape[2]
            W = rgb.shape[3]

            # train with DUT
            score1, score2, score3, score4, s1, s2, s3, s4, = net(rgb, 1)
            sal_loss = loss_1(score1, score2, score3, score4, s1, s2, s3, s4,  label)


            loss1 = sal_loss
            loss_sod_value += sal_loss.data


            # train with target

            score1_t, score2_t, score3_t, score4_t, s1_t, s2_t, s3_t, s4_t, = net(target_rgb, 2)
            sal_loss_t = loss_1(score1_t, score2_t, score3_t, score4_t, s1_t, s2_t, s3_t, s4_t,  target_label)


            loss2 = sal_loss_t
            loss_sodt_value += sal_loss_t.data


            loss = loss1 + loss2
            loss.backward()

            optimizer.step()
            net.zero_grad()

            if i % 10 == 0:
                print('epoch: [%2d/%2d], iter: [%5d/%5d] || loss_sod : %5.4f , loss_sodt : %5.4f| lr:%6.5f' % (
                    epochi, epoch, i, iter_num, loss_sod_value / 100, loss_sodt_value / 100, lr))



            rgb, label = prefetcher.next()
            target_rgb, target_label = target_prefetcher.next()


        if epochi >= 0 and epochi % 2 == 0:
            torch.save(net.state_dict(), '%s/epoch_%d.pth' % (save_path, epochi))
    torch.save(net.state_dict(), '%s/UMD.pth' % (save_path))