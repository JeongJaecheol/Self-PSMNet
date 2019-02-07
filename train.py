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
import torchvision.utils
from tensorboardX import SummaryWriter

from dataloader import listflowfile as lt    #from dataloader import listburyfile as lt
from dataloader import SceneFlowLoader as DA #from dataloader import MiddleburyLoader as DA
from utils import pfm_IO
from utils.demo import *
from models import *

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='stackhourglass_3d_share', # stackhourglass
                    help='select model')
parser.add_argument('--datapath', default='../Depth-Estimation/data/Scene Flow Datasets/', # ../Depth-Estimation/data/Middlebury Datasets/
                    help='datapath')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=12,
                    help='number of batch size to train')
parser.add_argument('--loadmodel', default= None, # './ckpt/checkpoint_10.tar'
                    help='load model')
parser.add_argument('--savemodel', default='./trained',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--gpuids', metavar='N', type=int, nargs='+', default=[0, 1, 2, 3],
                    help='enables CUDA')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

all_left_img, all_right_img, all_left_disp, all_right_disp, test_left_img, test_right_img, test_left_disp, test_right_disp = lt.dataloader(args.datapath)
#print(all_right_disp[0])
#print(test_right_disp[0])
#print(len(all_left_img), len(all_right_img), len(all_left_disp), len(test_left_img), len(test_right_img), len(test_left_disp))
TrainImgLoader = torch.utils.data.DataLoader(
         DA.myImageFloder(all_left_img,all_right_img,all_left_disp, all_right_disp, True), 
	     batch_size= args.batch_size, shuffle= True, num_workers= 12, drop_last=False)
TestImgLoader = torch.utils.data.DataLoader(
         DA.myImageFloder(test_left_img,test_right_img,test_left_disp, test_right_disp, False), 
         batch_size= 1, shuffle= False, num_workers= 1, drop_last=False)

if args.model == 'basic':
    model = basic(args.maxdisp)
if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'stackhourglass_self':
    model = stackhourglass_self(args.maxdisp)
else:
    print('no model')

if args.cuda:
    torch.cuda.set_device(args.gpuids[0])
    model = nn.DataParallel(model, device_ids=args.gpuids)
    model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

def train(imgL, imgR, disp_L, disp_R, t_iter = None, writer = None):
    model.train()
    imgL   = Variable(torch.FloatTensor(imgL))
    imgR   = Variable(torch.FloatTensor(imgR))   
    dispL_true = Variable(torch.FloatTensor(disp_L))
    dispR_true = Variable(torch.FloatTensor(disp_R))
    #if np.array_equal(dispL_true.numpy(), dispR_true.numpy()):
    #    print('same')
    #else:
    #    print('different')

    if args.cuda:
        imgL, imgR = imgL.cuda(), imgR.cuda()
        dispL_true, dispR_true = dispL_true.cuda(), dispR_true.cuda()

    #---------
    maskL = dispL_true < args.maxdisp
    maskL.detach_()
    maskR = dispR_true < args.maxdisp
    maskR.detach_()
    #----
    optimizer.zero_grad()
        
    if args.model == 'basic':
        output3 = model(imgL,imgR)
        output = torch.squeeze(output3,1)
        loss = F.smooth_l1_loss(output3[maskL], dispL_true[maskL], size_average=True)
    elif args.model == 'stackhourglass':
        output1, output2, output3 = model(imgL,imgR)
        output1 = torch.squeeze(output1,1)
        output2 = torch.squeeze(output2,1)
        output3 = torch.squeeze(output3,1)
        loss_1 = F.smooth_l1_loss(output1[maskL], dispL_true[maskL], size_average=True)
        loss_2 = F.smooth_l1_loss(output2[maskL], dispL_true[maskL], size_average=True)
        loss_3 = F.smooth_l1_loss(output3[maskL], dispL_true[maskL], size_average=True) 
        loss = 0.5*loss_1 + 0.7*loss_2 + loss_3
    elif args.model == 'stackhourglass_self':
        output, _ = model(imgL,imgR)
        loss = F.l1_loss(output, imgR, size_average=True)

    loss.backward()
    optimizer.step()

    if writer.__class__.__name__ is "SummaryWriter":
        if args.model == 'basic':
            for name, param in model.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), t_iter)
            writer.add_scalar(tag = 'loss/total_loss', scalar_value = loss.data[0], global_step = t_iter)
            if t_iter % 1000:
                writer.add_image('Left Input Image', torchvision.utils.make_grid(imgL, normalize=True, scale_each=True), t_iter)
                writer.add_image('Right Input Image', torchvision.utils.make_grid(imgR, normalize=True, scale_each=True), t_iter)
                writer.add_image('Ground Truth Left disparity map', torchvision.utils.make_grid(dispL_true, normalize=True, scale_each=True), t_iter)
                writer.add_image('Ground Truth Right disparity map', torchvision.utils.make_grid(dispR_true, normalize=True, scale_each=True), t_iter)
                writer.add_image('model Left disparity map', torchvision.utils.make_grid(output, normalize=True, scale_each=True), t_iter)
        elif args.model == 'stackhourglass':
            for name, param in model.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), t_iter)
            writer.add_scalar(tag = 'loss/loss_1', scalar_value = loss_1.data[0], global_step = t_iter)
            writer.add_scalar(tag = 'loss/loss_2', scalar_value = loss_2.data[0], global_step = t_iter)
            writer.add_scalar(tag = 'loss/loss_3', scalar_value = loss_3.data[0], global_step = t_iter)
            writer.add_scalar(tag = 'loss/total_loss', scalar_value = loss.data[0], global_step = t_iter)
            if t_iter % 1000:
                writer.add_image('Left Input Image', torchvision.utils.make_grid(imgL, normalize=True, scale_each=True), t_iter)
                writer.add_image('Right Input Image', torchvision.utils.make_grid(imgR, normalize=True, scale_each=True), t_iter)
                writer.add_image('Ground Truth Left disparity map', torchvision.utils.make_grid(dispL_true, normalize=True, scale_each=True), t_iter)
                writer.add_image('Ground Truth Right disparity map', torchvision.utils.make_grid(dispR_true, normalize=True, scale_each=True), t_iter)
                writer.add_image('model Left disparity map', torchvision.utils.make_grid(output3, normalize=True, scale_each=True), t_iter)
        elif args.model == 'stackhourglass_self':
            for name, param in model.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), t_iter)
            writer.add_scalar(tag = 'loss/total_loss', scalar_value = loss.data[0], global_step = t_iter)
            if t_iter % 1000:
                writer.add_image('Left Input Image', torchvision.utils.make_grid(imgL, normalize=True, scale_each=True), t_iter)
                writer.add_image('Right Input Image', torchvision.utils.make_grid(imgR, normalize=True, scale_each=True), t_iter)
                writer.add_image('Ground Truth Left disparity map', torchvision.utils.make_grid(dispL_true, normalize=True, scale_each=True), t_iter)
                writer.add_image('Ground Truth Right disparity map', torchvision.utils.make_grid(dispR_true, normalize=True, scale_each=True), t_iter)
                writer.add_image('model Left disparity map', torchvision.utils.make_grid(output, normalize=True, scale_each=True), t_iter)
    else:
        return loss.data[0]

def test(imgL,imgR,disp_true):
    model.eval()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))  
    if args.cuda:
        imgL, imgR = imgL.cuda(), imgR.cuda()

    #---------
    mask = disp_true < 192
    #----

    with torch.no_grad():
        output3 = model(imgL,imgR)

    output = torch.squeeze(output3.data.cpu(),1)[:,4:,:]

    if len(disp_true[mask])==0:
        loss = 0
    else:
        loss = torch.mean(torch.abs(output[mask]-disp_true[mask]))  # end-point-error

    return loss

def adjust_learning_rate(optimizer, epoch):
    lr = 0.001
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    writer = SummaryWriter(log_dir='./Tensorboard/%s/' %(args.model), comment='')
    writer.add_graph(model = model, input_to_model=None)
    
    start_full_time = time.time()
    total_iter = 0
    batch_idx = 0
    for epoch in range(1, args.epochs+1):
	    print('This is %d-th epoch' %(epoch))
	    total_train_loss = 0
	    adjust_learning_rate(optimizer,epoch)
	    ## training ##
	    for batch_idx, (imgL_crop, imgR_crop, disp_crop_L, disp_crop_R) in enumerate(TrainImgLoader, 1):
	        start_time = time.time()
	        loss = train(imgL_crop, imgR_crop, disp_crop_L, disp_crop_R, t_iter = total_iter + batch_idx, writer = writer)
	        print('Iter %d , training loss = %.3f , time = %.2f' %(batch_idx, loss, time.time() - start_time))
	        total_train_loss += loss      
        
	    print('----------------------------------------------------------------------') 
	    print('epoch %d total training loss = %.3f' %(epoch, total_train_loss/len(TrainImgLoader)))
	    total_iter = total_iter + batch_idx
	    print('1 epoch total iteration = %d' %(batch_idx))
	    print('total training iteration = %d' %(total_iter))
	    print('----------------------------------------------------------------------') 

	    #SAVE
	    savefilename = args.savemodel+'/non_pretrained_3d_share_ckpt_'+str(epoch)+'.tar'
	    torch.save({
		    'epoch': epoch,
		    'state_dict': model.state_dict(),
                    'train_loss': total_train_loss/len(TrainImgLoader),
		}, savefilename)

    print('full training time = %.2f HR' %((time.time() - start_full_time)/3600))
    
    writer.close()

    #------------- TEST ------------------------------------------------------------
	#----------------------------------------------------------------------------------

	#SAVE test information
    
	#savefilename = args.savemodel+'testinformation.tar'
	#torch.save({
	#    	'test_loss': total_test_loss/len(TestImgLoader),
	#	}, savefilename)
    


if __name__ == '__main__':
    main()
    