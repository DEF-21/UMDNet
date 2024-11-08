#coding=utf-8

import os
import cv2
import numpy as np
import torch
try:
    from . import transform
except:
    import transform
from torch.utils.data import Dataset
# BGR
# MSRA-B

# mean_rgb = np.array([[[0.485, 0.456, 0.406]]])*255
# mean_t =np.array([[[0.485, 0.456, 0.406]]])*255
# std_rgb = np.array([[[0.229, 0.224, 0.225]]])*255
# std_t = np.array([[[0.229, 0.224, 0.225]]])*255

# VT5000

mean_rgb = np.array([[[0.551, 0.619, 0.532]]])*255
# mean_t =np.array([[[0.341,  0.360, 0.753]]])*255
std_rgb = np.array([[[0.241, 0.236, 0.244]]])*255
# std_t = np.array([[[0.208, 0.269, 0.241]]])*255

# def getRandomSample(rgb,t):
#     n = np.random.randint(10)
#     zero = np.random.randint(2)
#     if n==1:
#         if zero:
#             rgb = torch.from_numpy(np.zeros_like(rgb))
#         else:
#             rgb = torch.from_numpy(np.random.randn(*rgb.shape))
#     elif n==2:
#         if zero:
#             t = torch.from_numpy(np.zeros_like(t))
#         else:
#             t = torch.from_numpy(np.random.randn(*t.shape))
#     return rgb,t

class Data(Dataset):
    def __init__(self, root,mode='train'):
        self.samples = []
        lines = os.listdir(os.path.join(root, 'GT'))
        self.mode = mode
        for line in lines:
            rgbpath = os.path.join(root, 'RGB', line[:-4]+'.jpg')
            # print(line + 'pri')
            maskpath = os.path.join(root, 'GT', line)
            self.samples.append([rgbpath, maskpath])

        if mode == 'train':
            self.transform = transform.Compose( transform.Normalize(mean1=mean_rgb, std1=std_rgb),
                                                transform.Resize(352, 352),
                                                transform.RandomHorizontalFlip(),
                                                transform.ToTensor())

        elif mode == 'test':
            self.transform = transform.Compose( transform.Normalize(mean1=mean_rgb, std1=std_rgb),
                                                transform.Resize(352, 352),
                                                transform.ToTensor())
        else:
            raise ValueError

    def __getitem__(self, idx):
        rgbpath, maskpath = self.samples[idx]
        # print(rgbpath)
        rgb = cv2.imread(rgbpath).astype(np.float32)
        # t = cv2.imread(tpath).astype(np.float32)
        mask = cv2.imread(maskpath).astype(np.float32)
        H, W, C = mask.shape
        rgb, mask = self.transform(rgb, mask)
        # if self.mode == 'train':
        #     rgb, t =getRandomSample(rgb, t)
        return rgb, mask, (H, W), maskpath.split('/')[-1]

    def __len__(self):
        return len(self.samples)
