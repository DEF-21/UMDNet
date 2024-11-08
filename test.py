import os
from torch.utils.data import DataLoader
from lib.dataset import Data
import torch.nn.functional as F
import torch
import cv2
import time
from UMDNet import Mnet
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if __name__ == '__main__':
    model_path = ''
    out_path = ''

    # data = Data(root='/home/ms/Documents/dataset/Metal_mix/NEU3/', mode='test')
    data = Data(root='', mode='test')

    loader = DataLoader(data, batch_size=1, shuffle=False)
    net = Mnet().cuda()
    print('loading model from %s...' % model_path)
    net.load_state_dict(torch.load(model_path))
    if not os.path.exists(out_path): os.mkdir(out_path)
    time_s = time.time()
    img_num = len(loader)
    net.eval()

    with torch.no_grad():
        for rgb, _, (H, W), name in loader:
            print(name[0])
            # score1, score2, score3, score4 = net(rgb.cuda().float())
            # score1, score2, score3, score4, s1, s2, s3, s4, tcp = net(rgb.cuda().float(), 2)
            score1, score2, score3, score4, s1, s2, s3, s4 = net(rgb.cuda().float(), 2)
            score = F.interpolate(score1, size=(H, W), mode='bilinear', align_corners=True)

            # score = F.interpolate(score1, size=(H, W), mode='bilinear', align_corners=True)
            pred = np.squeeze(torch.sigmoid(score).cpu().data.numpy())
            #pred[pred > 0.5] = 1
            # pred[pred < 1] = 0
            cv2.imwrite(os.path.join(out_path, name[0][:-4] + '.png'), 255 * pred)
            # cv2.imwrite(os.path.join(out_path, name[0][:-4] + '.png'), pred)

    time_e = time.time()
    print('speed: %f FPS' % (img_num / (time_e - time_s)))



