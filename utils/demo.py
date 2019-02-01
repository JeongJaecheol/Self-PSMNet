import cv2
import numpy as np
import torch
from torch.autograd import Variable

def demo(epoch, batch_idx, args, model):
    if batch_idx % 1000 == 0:
        model.eval()
        imgL = []
        l_img = cv2.imread('./demo/0000_L.png')
        l_img = cv2.resize(l_img, (512, 256), interpolation=cv2.INTER_CUBIC)
        l_img = np.transpose(l_img, (2, 0, 1))
        imgL.append(l_img)
        imgL = Variable(torch.FloatTensor(imgL))
        imgR = []
        r_img = cv2.imread('./demo/0000_R.png')
        r_img = cv2.resize(r_img, (512, 256), interpolation=cv2.INTER_CUBIC)
        r_img = np.transpose(r_img, (2, 0, 1))
        imgR.append(r_img)
        imgR   = Variable(torch.FloatTensor(imgR))
        if args.model == 'stackhourglass_3d_share':
            _, __, predic_L, _, __, predic_R = model(imgL, imgR)
            predic_L = torch.squeeze(predic_L.data.cpu(), 1)
            predic_R = torch.squeeze(predic_R.data.cpu(), 1)
            predic_L = np.array(predic_L)
            predic_R = np.array(predic_R)
            predic_L = np.transpose(predic_L, (1, 2, 0))
            predic_R = np.transpose(predic_R, (1, 2, 0))
            output_img = cv2.resize(output_img, (960, 540), interpolation=cv2.INTER_CUBIC) * 960 / 512
            output_img = cv2.applyColorMap(output_img.astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite('./demo/result_L_epoch_{0}iter_{1}.png'.format(epoch, batch_idx), output_img)
        return 1
    else:
        return 0
'''
def demo(epoch, batch_idx, args, model):
    if batch_idx % 1000 == 0:
        model.eval()
        imgL = []
        l_img = cv2.imread('./demo/0000_L.png')
        l_img = cv2.resize(l_img, (512, 256), interpolation=cv2.INTER_CUBIC)
        l_img = np.transpose(l_img, (2, 0, 1))
        imgL.append(l_img)
        imgL = Variable(torch.FloatTensor(imgL))
        imgR = []
        r_img = cv2.imread('./demo/0000_R.png')
        r_img = cv2.resize(r_img, (512, 256), interpolation=cv2.INTER_CUBIC)
        r_img = np.transpose(r_img, (2, 0, 1))
        imgR.append(r_img)
        imgR   = Variable(torch.FloatTensor(imgR))
        if args.model == 'stackhourglass_3d_share':
            _, __, right_ = model(imgL, imgR)
            _, __, predic_R = model(imgL, imgR)
            imgR = torch.squeeze(imgR.data.cpu(), 0)
            predic_R = torch.squeeze(predic_R.data.cpu(), 1)
            output_img = predic_R
            output_img = np.array(output_img)
            output_img = np.transpose(output_img, (1, 2, 0))
            output_img = cv2.resize(output_img, (960, 540), interpolation=cv2.INTER_CUBIC) * 960 / 512
            output_img = cv2.applyColorMap(output_img.astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite('./demo/result_L_epoch_{0}iter_{1}.png'.format(epoch, batch_idx), output_img)
        return 1
    else:
        return 0
'''