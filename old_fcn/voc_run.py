import matplotlib.pyplot as plt
import torch
import os
import pdb
import gc
import PIL.Image as Image
import torch.nn.functional as F
import torchvision
import numpy as np
from torchvision.transforms import Compose, Normalize, ToTensor
from torch.autograd.variable import Variable
from torch.utils import data
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from datetime import datetime

from dataset import VOCDataSet
from transform import ReLabel, ToLabel, ToSP, Scale, Colorize
from utils import make_image_grid, make_label_grid, CrossEntropyLoss2d, compute_mean_iou, FCN_metric
from resnet import resnet101, resnet50
from model import Seg

# torch.cuda.set_device(2)
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['CUDA_VISIBLE_DEVICES']='2,3'

# os.system('rm -rf ./runs/*')
writer = SummaryWriter('runs/'+datetime.now().strftime('VOC_%B%d  %H:%M:%S'))

std = [.229, .224, .225]
mean = [.485, .456, .406]

input_transform = Compose([
    Scale((256, 256), Image.BILINEAR),
    ToTensor(),
    Normalize(mean, std),
])

target_transform = Compose([
    Scale((256, 256), Image.NEAREST),
    ToSP(256),
    ToLabel(),
    ReLabel(255, 21),
])


loader = data.DataLoader(VOCDataSet("../datasets/VOC/VOCdevkit/VOC2012/",
                                    img_transform=input_transform,
                                    label_transform=target_transform),
                                    batch_size=32, shuffle=True, pin_memory=True)

# res101 = resnet101(pretrained=True).cuda()

res50 = resnet50(pretrained=True).cuda()

seg = Seg().cuda()

weight = torch.ones(22)
weight[21] = 0

criterion = CrossEntropyLoss2d(weight.cuda())

optimizer_seg = torch.optim.Adam(seg.parameters(), lr=1e-3)
optimizer_feat = torch.optim.Adam(res50.parameters(), lr=1e-4)

for t in range(50):
    for i, (img, label) in enumerate(loader):
        img = img.cuda()
        label = label[0].cuda()
        label = Variable(label)
        inputs = Variable(img)

        feats = res50(inputs)
        output = seg(feats)

        seg.zero_grad()
        res50.zero_grad()
        loss = criterion(output, label)
        loss.backward()
        optimizer_feat.step()
        optimizer_seg.step()

        ## see
        inputs = make_image_grid(img, mean, std)
        label = make_label_grid(label.data)
        label = Colorize()(label).type(torch.FloatTensor)
        output = make_label_grid(torch.max(output, dim=1)[1].data)
        output = Colorize()(output).type(torch.FloatTensor)
        writer.add_image('image', inputs, i)
        writer.add_image('label', label, i)
        writer.add_image('pred', output, i)
        writer.add_scalar('loss', loss.data[0], i)
        metric = FCN_metric(output, label)
        writer.add_scalar('mIU',metric['MIU'],i)
#         writer.add_scalar('iou', compute_mean_iou(output,label),i)
        if i % 100 is 0:
            print(output.shape)
            # plt.imshow(np.asarray(output))
#             plt.show()
        print("epoch %d step %d, loss=%.4f, %s" %(t, i, loss.data.cpu()[0], metric))