import os.path as osp

import fcn

import torch
import torchvision

data_dir = 'models/pytorch/'

def VGG16(pretrained=False):
    model = torchvision.models.vgg16(pretrained=False)
    if not pretrained:
        return model
    model_file = _get_pretrained()
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    return model


# catched_download() will return path if path exists


def _get_vgg16_pretrained_model():
    return fcn.data.cached_download(
        url='http://drive.google.com/uc?id=0B9P1L--7Wd2vLTJZMXpIRkVVRFk',
        path=osp.expanduser(data_dir+'vgg16_from_caffe.pth'),
        md5='aa75b158f4181e7f6230029eb96c1b13',
    )


def _get_pretrained():
    return osp.expanduser(data_dir+'vgg16_from_caffe.pth')
