import numpy as np
import os
from metric_data import test_dataset
from saliency_metric import cal_mae,cal_fm,cal_sm,cal_em,cal_wfm,MissedDetectionRate,FalsePositiveRate


sal_root = ''

gt_root = ''

test_loader = test_dataset(sal_root, gt_root)
mae,fm,sm,em,wfm,mdr,fpr= cal_mae(),cal_fm(test_loader.size),cal_sm(),cal_em(),cal_wfm(),MissedDetectionRate(),FalsePositiveRate()
for i in range(test_loader.size):
    print ('predicting for %d / %d' % ( i + 1, test_loader.size))
    sal, gt = test_loader.load_data()
    if sal.size != gt.size:
        x, y = gt.size
        sal = sal.resize((x, y))
    gt = np.asarray(gt, np.float32)
    gt /= (gt.max() + 1e-8)
    gt[gt > 0.5] = 1
    gt[gt != 1] = 0
    res = sal
    res = np.array(res)
    if res.max() == res.min():
        res = res/255
    else:
        res = (res - res.min()) / (res.max() - res.min())
    sal_bin = (res > 0.5).astype(np.uint8)
    gt_bin = (gt > 0.5).astype(np.uint8)


    TP = np.sum((sal_bin == 1) & (gt_bin == 1))
    FP = np.sum((sal_bin == 1) & (gt_bin == 0))
    TN = np.sum((sal_bin == 0) & (gt_bin == 0))
    FN = np.sum((sal_bin == 0) & (gt_bin == 1))

    mae.update(res, gt)
    sm.update(res,gt)
    fm.update(res, gt)
    em.update(res,gt)
    wfm.update(res,gt)
    mdr.update(TP, FN)
    fpr.update(FP,TN)
MAE = mae.show()
maxf,meanf,_,_ = fm.show()
sm = sm.show()
em = em.show()
wfm = wfm.show()
MDR = mdr.show()
FPR = fpr.show()
print('MAE: {:.3f} maxF: {:.3f} avgF: {:.3f} wfm: {:.3f} Sm: {:.3f} Em: {:.3f} MDR: {:.3f} FPR: {:.3f}'.format(
    MAE, maxf, meanf, wfm, sm, em, MDR, FPR))
