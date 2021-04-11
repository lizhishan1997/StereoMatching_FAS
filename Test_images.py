import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import time
import cv2

from StereoMatching_Net import StereoMatching_Net
from Classification_Net import ClassificationNet


parser = argparse.ArgumentParser(description='PSMNet')

parser.add_argument('--load_stereo', default='weights/stereo.pth',
                    help='loading stereo model')
parser.add_argument('--load_classfy', default='weights/classification.pth',
                    help='loading classify model')
parser.add_argument('--leftimg', default= 'test_images/spoofing_left.jpg',
                    help='load model')
parser.add_argument('--rightimg', default= 'test_images/spoofing_right.jpg',
                    help='load model')
parser.add_argument('--maxdisp', type=int, default=24,
                    help='maxium disparity')

args = parser.parse_args()


Stereo_Part=StereoMatching_Net(maxdisp=args.maxdisp)
Classfy_Part=ClassificationNet()

Stereo_Part.load_state_dict({k.replace('module.',''):v for k,v in torch.load(args.load_stereo,map_location='cpu')['state_dict'].items()})
Classfy_Part.load_state_dict({k.replace('module.',''):v for k,v in torch.load(args.load_classfy,map_location='cpu').items()})



def test(imgL,imgR):
        Stereo_Part.eval()
        Classfy_Part.eval()

        with torch.no_grad():
            disp = Stereo_Part(imgL, imgR)
            output = Classfy_Part(disp.unsqueeze(0))
            output=output.softmax(dim=1)
            if output.argmax(dim=1).item()==0:
                print('Living, confidence is %.2f%%'%(100*output.max()))
            else:
                print('Spoofing, confidence is %.2f%%' % (100 * output.max()))

        disp = torch.squeeze(disp)
        pred_disp = disp.data.cpu().numpy()

        return pred_disp


def main():

        infer_transform = transforms.Compose([transforms.ToTensor(),
                                              ])
        imgL=cv2.imread(args.leftimg,cv2.IMREAD_UNCHANGED)
        imgR=cv2.imread(args.rightimg,cv2.IMREAD_UNCHANGED)
        imgL=infer_transform(imgL)
        imgR=infer_transform(imgR)
        imgL=imgL.unsqueeze(0)
        imgR=imgR.unsqueeze(0)

        start_time = time.time()
        pred_disp = test(imgL,imgR)
        print('Time consuming = %.2f s' %(time.time() - start_time))
        img=pred_disp

        img = (img*10).astype('uint8')

        cv2.imwrite('prediction.png',img)


if __name__ == '__main__':
   main()
