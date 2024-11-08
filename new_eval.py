"""
https://blog.csdn.net/sinat_29047129/article/details/103642140
https://www.cnblogs.com/Trevo/p/11795503.html
refer to https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/metrics.py
"""
import numpy as np
import os
from PIL import Image

__all__ = ['SegmentationMetric']

"""
confusionMetric
P\L     P    N
P      TP    FP
N      FN    TN
"""


class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)  # 混淆矩阵n*n，初始值全0

    # 像素准确率PA，预测正确的像素/总像素
    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        # acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    # 类别像素准确率CPA，返回n*1的值，代表每一类，包括背景
    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / np.maximum(self.confusionMatrix.sum(axis=1), 1)
        return classAcc

    # 类别像素召回率
    def classRecall(self):
        # A more accurate way to call it recall
        # recall = (TP) / TP + FN
        recall = np.diag(self.confusionMatrix) / np.maximum(self.confusionMatrix.sum(axis=0), 1)
        return recall

    # 类别平均像素准确率MPA，对每一类的像素准确率求平均
    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)
        return meanAcc

    # 类别平均像素召回率，对每一类的像素召回率求平均
    def meanrecall(self):
        recall = self.classRecall()
        meanre = np.nanmean(recall)
        return meanre

    # Fmeasure
    def Fmeasure(self):
        pre = self.meanPixelAccuracy()
        recall = self.meanrecall()
        return 2 * pre * recall / (pre + recall)

    # MIoU
    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)
        union = np.maximum(np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix), 1)
        IoU = intersection / union
        mIoU = np.nanmean(IoU)
        return mIoU

    # 根据标签和预测图片返回其混淆矩阵
    def genConfusionMatrix(self, imgPredict, imgLabel):
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask].astype(int) + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusionMatrix, axis=1) / np.sum(self.confusionMatrix)
        iu = np.diag(self.confusionMatrix) / (
                np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) -
                np.diag(self.confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    # 更新混淆矩阵
    def addBatch(self, imgPredict, imgLabel):
        # print(imgPredict.shape, imgLabel.shape)
        # print(imgPredict)
        assert imgPredict.shape == imgLabel.shape  # 确认标签和预测值图片大小相等

        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    # 清空混淆矩阵
    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))


def old():
    imgPredict = np.array([0, 0, 0, 1, 2, 2])
    imgLabel = np.array([0, 0, 1, 1, 2, 2])
    metric = SegmentationMetric(2)
    metric.addBatch(imgPredict, imgLabel)
    acc = metric.pixelAccuracy()
    macc = metric.meanPixelAccuracy()
    mIoU = metric.meanIntersectionOverUnion()
    print(acc, macc, mIoU)


def evaluate1(pre_path, label_path):
    acc_list = []
    pre_list = []
    mIoU_list = []
    fwIoU_list = []
    recall_list = []
    fmeasure_list = []

    pre_imgs = os.listdir(pre_path)
    lab_imgs = os.listdir(label_path)

    for i, p in enumerate(pre_imgs):
        imgPredict = Image.open(pre_path + p)
        imgPredict = np.array(imgPredict)
        # imgPredict = imgPredict[:,:,0]
        imgLabel = Image.open(label_path + lab_imgs[i])
        imgLabel = np.array(imgLabel)
        # imgLabel = imgLabel[:,:,0]

        metric = SegmentationMetric(2)  # 表示分类个数，包括背景
        metric.addBatch(imgPredict, imgLabel)
        acc = metric.pixelAccuracy()
        pre = metric.meanPixelAccuracy()
        mIoU = metric.meanIntersectionOverUnion()
        fwIoU = metric.Frequency_Weighted_Intersection_over_Union()
        recall = metric.meanrecall()
        fmeasure = metric.Fmeasure()

        acc_list.append(acc)
        pre_list.append(pre)
        mIoU_list.append(mIoU)
        fwIoU_list.append(fwIoU)
        recall_list.append(recall)
        fmeasure_list.append(fmeasure)

        # print('{}: acc={}, pre={}, mIoU={}, f1={}'.format(p, acc, pre, mIoU, fmeasure))

    return acc_list, pre_list, mIoU_list, fwIoU_list, recall_list, fmeasure_list


def evaluate2(pre_path, label_path):
    pre_imgs = os.listdir(pre_path)
    lab_imgs = os.listdir(label_path)

    metric = SegmentationMetric(2)  # 表示分类个数，包括背景
    for i, p in enumerate(pre_imgs):
        imgPredict = Image.open(pre_path + p)
        imgPredict = np.array(imgPredict)
        imgLabel = Image.open(label_path + lab_imgs[i])
        imgLabel = np.array(imgLabel)

        metric.addBatch(imgPredict, imgLabel)

    return metric


if __name__ == '__main__':
    pre_path = ''
    # pre_path = '/home/ms/PycharmProjects/mmsegmentation-0.14.0/tools/data/steel/GT0/'

    # label_path = '/home/ms/Documents/dataset/Metal_div/Type_I/test/GT0/'
    label_path = ''


    # 计算测试集每张图片的各种评价指标，最后求平均
    acc_list, pre_list, mIoU_list, fwIoU_list, recall_list, fmeasure_list = evaluate1(pre_path, label_path)
    print('final1: acc={:.2f}%, pre={:.2f}%, mIoU={:.2f}%, fwIoU={:.2f}%, recall={:.2f}%, fmeasure={:.2f}%'
          .format(np.mean(acc_list) * 100, np.mean(pre_list) * 100,
                  np.mean(mIoU_list) * 100, np.mean(fwIoU_list) * 100,
                  np.mean(recall_list) * 100, np.mean(fmeasure_list) * 100))

  #  # 加总测试集每张图片的混淆矩阵，对最终形成的这一个矩阵计算各种评价指标
    metric = evaluate2(pre_path, label_path)
    acc = metric.pixelAccuracy()
    pre = metric.meanPixelAccuracy()
    mIoU = metric.meanIntersectionOverUnion()
    fwIoU = metric.Frequency_Weighted_Intersection_over_Union()
    recall = metric.meanrecall()
    fmeasure = metric.Fmeasure()
    print('final2: acc={:.2f}%, pre={:.2f}%, mIoU={:.2f}%, fwIoU={:.2f}%, recall={:.2f}%, fmeasure={:.2f}%'
          .format(acc * 100, pre * 100, mIoU * 100, fwIoU * 100, recall * 100, fmeasure * 100))
#
