from __future__ import print_function, division

import sys

sys.path.append('..')
import argparse
import time
import os
import scipy.io
import yaml
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms

from DGnet.reIDmodel import ft_net, ft_netAB, ft_net_dense, PCB, PCB_test

# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

######################################################################
# Options
# --------
'''
action='store_true' 表示由传参时为True，否则为False
如python a.py -multi ，则会使用multi选项
python a.py ，则不会
'''
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='0,1,2', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch', default=100000, type=int, help='80000')
parser.add_argument('--test_dir', default='../Market/pytorch', type=str, help='./test_data')
parser.add_argument('--name', default='E0.5new_reid0.5_w30000', type=str, help='save model path')
parser.add_argument('--batchsize', default=80, type=int, help='batchsize')
parser.add_argument('--use_dense', action='store_true', help='use densenet121')
parser.add_argument('--PCB', action='store_true', help='use PCB')
parser.add_argument('--multi', default=False, help='use multiple query', )

opt = parser.parse_args()

str_ids = opt.gpu_ids.split(',')
which_epoch = opt.which_epoch
name = opt.name
data_dir = opt.test_dir
batchsize = opt.batchsize
gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        gpu_ids.append(id)

# set gpu ids
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
data_transforms = transforms.Compose([
    #transforms.Resize((256, 128), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if opt.PCB:
    data_transforms = transforms.Compose([
        transforms.Resize((384, 192), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
use_gpu = torch.cuda.is_available()


######################################################################
# Load model
# ---------------------------
def load_network(network):
    save_path = os.path.join('/home/fudan/Desktop/zls/tracking_pj/tracking/reid/outputs',name,'checkpoints/id_%08d.pt'%opt.which_epoch)
    
    state_dict = torch.load(save_path)
    network.load_state_dict(state_dict['a'], strict=False)
    return network


######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal水平翻转'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def norm(f):
    # f = f.squeeze()

    fnorm = torch.norm(f, p=2, dim=1, keepdim=True)
    f = f.div(fnorm.expand_as(f))
    return f


###################################################################

# ######################################################################
# Load Collected data Trained model

###load config###
config_path = os.path.join('/home/fudan/Desktop/zls/tracking_pj/tracking/reid/outputs',name,'config.yaml')
with open(config_path, 'r') as stream:
    config = yaml.load(stream)

model_structure = ft_netAB(config['ID_class'], norm=config['norm_id'], stride=config['ID_stride'], pool=config['pool'])

if opt.PCB:
    model_structure = PCB(config['ID_class'])

model = load_network(model_structure)

# Remove the final fc layer and classifier layer
model.model.fc = nn.Sequential()
model.classifier1.classifier = nn.Sequential()
model.classifier2.classifier = nn.Sequential()

# Change to test mode
model = model.eval()
if use_gpu:
    model = model.cuda()


class ReID:
    def __init__(self, is_folder=False):
        self.is_folder = is_folder
        self.data_transforms = data_transforms

    def extract_feature(self, features, data):
        if self.is_folder:
            img, label = data
        else:
            img = data
        # 载入图像数据到模型
        n, c, h, w = img.size()
        if opt.use_dense:  # False
            ff = torch.FloatTensor(n, 1024).zero_()
        else:
            ff = torch.FloatTensor(n, 1024).zero_()
        if opt.PCB:  # False
            ff = torch.FloatTensor(n, 2048, 6).zero_()  # we have six parts
        for i in range(2):
            if (i == 1):
                img = fliplr(img)
            # 输入到模型中
            input_img = Variable(img.cuda())
            f, x = model(input_img)
            x[0] = norm(x[0])
            x[1] = norm(x[1])
            f = torch.cat((x[0], x[1]), dim=1)  # use 512-dim feature
            f = f.data.cpu()
            ff = ff + f

        ff[:, 0:512] = norm(ff[:, 0:512])
        ff[:, 512:1024] = norm(ff[:, 512:1024])

        # norm feature
        if opt.PCB:
            # feature size (n,2048,6)
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6)
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        features = torch.cat((features, ff), 0)
        return features

    def get_feature(self, img): # img是列表，里面存放若干个PIL格式的RoI
        features = torch.FloatTensor()
        if self.is_folder:
            image_datasets = datasets.ImageFolder(os.path.join(img), self.data_transforms)
            dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=batchsize,
                                                      shuffle=False, num_workers=16)
            for data in dataloaders:
                features = self.extract_feature(features, data)
        else:
            # print(img.shape)
            img = [self.data_transforms(img[i])[np.newaxis, :] for i in range(len(img))]

            # 维度顺序为：n * c * h * w，如果不是，要调换
            img = np.concatenate([img[i] for i in range(len(img))], axis=0)
            img = torch.FloatTensor(img)  # img.type(torch.FloatTensor)

            features = self.extract_feature(features, img)

        return features
