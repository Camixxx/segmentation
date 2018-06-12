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


## iou
def compute_mean_iou(pred, label):
    pred = pred.data.numpy()
    label = label.data.numpy()
    unique_labels = np.unique(label)
    num_unique_labels = len(unique_labels);

    I = np.zeros(num_unique_labels)
    U = np.zeros(num_unique_labels)
    for index, val in enumerate(unique_labels):
        pred_i = pred == val
        label_i = label == val

        I[index] = float(np.sum(np.logical_and(label_i, pred_i)))
        U[index] = float(np.sum(np.logical_or(label_i, pred_i)))

    # mean_iou = np.mean(I / U)
    return np.mean(I / U)


# Recommend
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)

