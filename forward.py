from __future__ import print_function
import argparse
import os
import random
import numpy as np
import time
import math
import cv2

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F

from dataloader import listflowfile as lt
from dataloader import SecenFlowLoader as DA
from models import *
from utils import pfm_IO

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='stackhourglass', # stackhourglass
                    help='select model')
parser.add_argument('--forwardpath', default=r'./forward', # ./ETRI_forward, ./forward
                    help='datapath')
parser.add_argument('--loadmodel', default= r'./trained/non_pretrained_3d_share_ckpt_10.tar', # ./trained/middlebury_ckpt_10000.tar, ./ckpt/checkpoint_10.tar, ./trained/self_ckpt_10000.tar ./trained/non_pretrained_3d_share_ckpt_10.tar
                    help='load model')
parser.add_argument('--resize', metavar='N', type=int, nargs='+', default=[960, 512],
                    help='size tuple(width, height) for resizing')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--gpuids', metavar='N', type=int, nargs='+', default=[0],
                    help='enables CUDA')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.model == 'basic':
    model = basic(args.maxdisp)
if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'stackhourglass_self':
    model = stackhourglass_self(args.maxdisp)

if args.cuda:
    torch.cuda.set_device(args.gpuids[0])
    model = nn.DataParallel(model, device_ids=args.gpuids)
    model.cuda()
state_dict = torch.load(args.loadmodel)
model.load_state_dict(state_dict['state_dict'])
print('Pretrained model loaded')

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

def main():
    L_paths = []
    for (path, dir, files) in os.walk(args.forwardpath):
        for filename in files:
            if "left" in path + "/" + filename and "disp_result" not in path + "/" + filename:
                filepath = path + "/" + filename
                L_paths.append(filepath.replace("//", "/"))

    for test_data_path in L_paths:
        imgL = []
        imgR = []
        left = cv2.imread(test_data_path.replace("//", "/"))
        right = cv2.imread(test_data_path.replace("//", "/").replace("left", "right"))
        width = left.shape[1]
        height = left.shape[0]
        left = cv2.resize(left, tuple(args.resize), interpolation=cv2.INTER_CUBIC)
        right = cv2.resize(right, tuple(args.resize), interpolation=cv2.INTER_CUBIC)
        left = np.transpose(left, (2, 0, 1))
        right = np.transpose(right, (2, 0, 1))
	
        imgL.append(left)
        imgR.append(right)

        imgL = Variable(torch.FloatTensor(imgL))
        imgR = Variable(torch.FloatTensor(imgR))

        if args.cuda:
            imgL, imgR = imgL.cuda(), imgR.cuda()

        with torch.no_grad():
            if args.model == 'basic':
                out1 = model(imgL, imgR)
                output = torch.squeeze(out1[0].data.cpu(), 1)[:, 4:, :]
                output_img = output
                output_img = np.array(output_img)
                output_img = np.transpose(output_img, (1, 2, 0))
                output_img = cv2.resize(output_img, (width, height), interpolation=cv2.INTER_CUBIC) * width / args.resize[0]
                pfm_IO.write(test_data_path.replace("//", "/").replace("left", "disp_result").replace(".png", ".pfm"), output_img)
                output_img = cv2.applyColorMap(output_img.astype(np.uint8), cv2.COLORMAP_JET)
                cv2.imwrite(test_data_path.replace("//", "/").replace("left", "disp_result"), output_img)
            elif args.model == 'stackhourglass':
                out1 = model(imgL, imgR)
                output = torch.squeeze(out1[0].data.cpu(), 1)[:, 4:, :]
                output_img = output
                output_img = np.array(output_img)
                output_img = np.transpose(output_img, (1, 2, 0))
                output_img = cv2.resize(output_img, (width, height), interpolation=cv2.INTER_CUBIC) * width / args.resize[0]
                pfm_IO.write(test_data_path.replace("//", "/").replace("left", "disp_result").replace(".png", ".pfm"), output_img)
                output_img = cv2.applyColorMap(output_img.astype(np.uint8), cv2.COLORMAP_JET)
                cv2.imwrite(test_data_path.replace("//", "/").replace("left", "disp_result"), output_img)
            elif args.model == 'stackhourglass_self':
                out1, out2 = model(imgL, imgR)
                output = torch.squeeze(out2.data.cpu(), 1)
                output_img = output
                output_img = np.array(output_img)
                output_img = np.transpose(output_img, (1, 2, 0))
                output_img = cv2.resize(output_img, (width, height), interpolation=cv2.INTER_CUBIC) * width / args.resize[0]
                pfm_IO.write(test_data_path.replace("//", "/").replace("left", "disp_result").replace(".png", ".pfm"), output_img)
                output_img = cv2.applyColorMap(output_img.astype(np.uint8), cv2.COLORMAP_JET)
                cv2.imwrite(test_data_path.replace("//", "/").replace("left", "disp_result"), output_img)

if __name__ == '__main__':
    main()
