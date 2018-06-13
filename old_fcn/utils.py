from torchvision.utils import make_grid
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


def make_image_grid(img, mean, std):
    img = make_grid(img)
    for i in range(3):
        img[i] *= std[i]
        img[i] += mean[i]
    return img


def make_label_grid(label):
    label = make_grid(label.unsqueeze(1).expand(-1, 3, -1, -1))[0:1]
    return label

# def get_pic(image):
#     if isinstance(image, str):
#         return Image.open(image)
#     return image

# def get_classes(pic):
#     classes = []
#     color_list = pic.getcolors()
#     for color in range(len(color_list)):
#         classes.append(color_list[color][1])
#     return classes

## iou
def compute_mean_iou(pred, label):
    pred = pred.data.numpy()
    label = label.data.numpy()
    unique_labels = np.unique(label)
    num_unique_labels = len(unique_labels)

    I = np.zeros(num_unique_labels)
    U = np.zeros(num_unique_labels)
    for index, val in enumerate(unique_labels):
        pred_i = pred == val
        label_i = label == val
        
        I[index] = float(np.sum(np.logical_and(label_i, pred_i)))
        U[index] = float(np.sum(np.logical_or(label_i, pred_i)))

    # mean_iou = np.mean(I / U)
    return np.mean(I / U)


def get_IU(pred, label):
    pred = pred.data.numpy()
    label = label.data.numpy()
    unique_labels = np.unique(label)
    num_unique_labels = len(unique_labels)

    I = np.zeros(num_unique_labels)
    U = np.zeros(num_unique_labels)
    for index, val in enumerate(unique_labels):
        pred_i = pred == val
        label_i = label == val
        I[index] = float(np.sum(np.logical_and(label_i, pred_i)))
        U[index] = float(np.sum(np.logical_or(label_i, pred_i)))
    return I,U


def get_total(pred, label):
    lb = label.data.numpy()
    pd = pred.data.numpy()
    total = pd == lb
    return total
     
    
def get_union(pred, label):
    pred = pred.data.numpy()
    label = label.data.numpy()
    unique_labels = np.unique(label)
    classes_num = len(unique_labels)
    unions = np.zeros(classes_num)
    for index, val in enumerate(unique_labels):
        pred_i = pred == val
        label_i = label == val
        unions[index] = float(np.sum(np.logical_or(label_i, pred_i)))
    return unions


def pixel_accuarcy(label, pred):
    t =  get_total(pred, label)
    pa = float(np.sum(t))/t.size
    return pa


def mean_pixel_accuracy(label, pred):
    I,U = get_IU(pred, label)
    label = label.data.numpy()
    unique_labels = np.unique(label)
    num_unique_labels = len(unique_labels)
    T = np.zeros(num_unique_labels)
    mpa = np.sum(I/U)/I.size
    return mpa

    
def mean_IU(label, pred):
    I,U = get_IU(pred, label)
    unions = get_union(pred, label)
    mIoU = float(np.sum(I/U))/I.size
    return mIoU


def frequency_weighted_IU(label, pred):
    I,U = get_IU(pred, label)
    total = get_total(pred, label)
    unions = get_union(pred, label)
    s = float(np.sum(total*I/U))
    FWIoU = s/I.size
    return FWIoU


def FCN_metric(label, pred):
    PA = pixel_accuarcy(label, pred)
    MPA = mean_pixel_accuracy(label, pred)
    MIU = mean_IU(label, pred)
#     FWIoU = frequency_weighted_IU(label, pred)
    result = {'PA':PA, 'MPA':MPA, 'MIU':MIU} #, 'FWIoU':FWIoU
    return result


#def test():
#   dir1 = 'D:/9527/2018.4.12 语义分割评估（未完成）/DJI_0605.png'
#   dir2 = 'D:/9527/2018.4.12 语义分割评估（未完成）/pred_9.png'
#   dir3 = 'D:/FCN.tensorflow-master-123/test2018.4.26/A/gt/gt_5.png'
#   dir4 = 'D:/FCN.tensorflow-master-123/test2018.4.26/A/pred/pred_5.png'
#   print('PA = ',pixel_accuarcy(dir1, dir2))
#   print('MPA = ',mean_pixel_accuracy(dir3, dir4))
#   print('MIoU = ',mean_IU(dir3, dir4))
#   print('FWIoU = ',frequency_weighted_IU(dir1, dir2))

def FCN_evaluate(dir1):
    PA = 0
    MPA = 0
    MIU = 0
    FWIoU = 0
    print('the directory is:',dir1)
    image_lists = create_image_lists(dir1)

    for i in range(len(image_lists['gt'])):
        print(i)
        ground_truth = dir1 +'/gt/'+image_lists['gt'][i]
        prediction = dir1 + '/pred/' + image_lists['pred'][i]
    result = {'name': name, 'PA':PA, 'MPA':MPA, 'MIU':MIU, 'FWIoU':FWIoU}
    return result

# Recommend
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)

