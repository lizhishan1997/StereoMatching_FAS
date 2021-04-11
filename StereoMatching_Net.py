import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


class disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super(disparityregression, self).__init__()
        self.disp = Variable(torch.Tensor(np.reshape(np.array(range(maxdisp)),[1,maxdisp,1,1])), requires_grad=False)

    def forward(self, x):
        disp = self.disp.repeat(x.size()[0],1,x.size()[2],x.size()[3])
        out = torch.sum(x*disp,1)
        return out

class StereoMatching_Net(nn.Module):
    def __init__(self, maxdisp):
        super(StereoMatching_Net, self).__init__()
        self.maxdisp = maxdisp

        self.feature=nn.Sequential(
            nn.Conv2d(48, 48, 5, 1, 2, groups=1, bias=False),
            nn.BatchNorm2d(48),
            nn.PReLU(),
            nn.Conv2d(48, 24, 3, 1, 1, bias=False),
            nn.BatchNorm2d(24),
            nn.PReLU(),
            nn.Conv2d(24, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, 48, 3, 2, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.PReLU(),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.PReLU()
        )

        self.SPP1_branch1=nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=24, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(in_channels=24, out_channels=24, kernel_size=4, stride=2, padding=1, groups=1,
                               bias=False)
        )

        self.SPP1_branch2=nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=24, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.ConvTranspose2d(in_channels=24, out_channels=24, kernel_size=4, stride=2, padding=1,
                               groups=1, bias=False),
            nn.ConvTranspose2d(in_channels=24, out_channels=24, kernel_size=4, stride=2, padding=1,
                               groups=1, bias=False)
        )

        self.SPP1_branch3=nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=24, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=8, stride=8),
            nn.ConvTranspose2d(in_channels=24, out_channels=24, kernel_size=4, stride=2, padding=1,
                               groups=1, bias=False),
            nn.ConvTranspose2d(in_channels=24, out_channels=24, kernel_size=4, stride=2, padding=1,
                               groups=1, bias=False),
            nn.ConvTranspose2d(in_channels=24, out_channels=24, kernel_size=4, stride=2, padding=1,
                               groups=1, bias=False)
        )

        self.SPP1_branch4=nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=24, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.PReLU()
        )

        self.aggregation=nn.Sequential(nn.Conv2d(in_channels=96, out_channels=48, kernel_size=3, padding=1, stride=1, bias=False),
                                       nn.BatchNorm2d(48),
                                       nn.PReLU(),
                                       nn.ConvTranspose2d(in_channels=48, out_channels=48, groups=1, kernel_size=4,
                                                          padding=1, stride=2, bias=False),
                                       nn.PReLU(),
                                       nn.Conv2d(in_channels=48, out_channels=24, kernel_size=3, padding=1, stride=1,
                                                 bias=False),
                                       nn.BatchNorm2d(24),
                                       nn.PReLU(),
                                       nn.ConvTranspose2d(in_channels=24, out_channels=24, groups=1, kernel_size=4,
                                                          padding=1, stride=2,bias=False),
                                       nn.PReLU()
                                       )

    def forward(self, left, right):

        cost=Variable(torch.FloatTensor(left.size()[0],left.size()[1]*2,int(self.maxdisp),left.size()[2],left.size()[3]).zero_())
        for i in range(int(self.maxdisp)):
            if i > 0:
                cost[:, :left.size()[1], i, :, i:] = left[:, :, :, i:]
                cost[:, left.size()[1]:, i, :, i:] = right[:, :, :, :-i]
            else:
                cost[:, :left.size()[1], i, :, :] = left
                cost[:, left.size()[1]:, i, :, :] = right
        cost = cost.contiguous().view(left.size()[0], left.size()[1] * 2 * int(self.maxdisp),
                                      left.size()[2], left.size()[3])

        x=self.feature(cost)
        branch1=self.SPP1_branch1(x)
        branch2=self.SPP1_branch2(x)
        branch3=self.SPP1_branch3(x)
        branch4=self.SPP1_branch4(x)
        x=torch.cat([branch1,branch2,branch3,branch4],dim=1)
        x=self.aggregation(x)
        x=F.softmax(x,dim=1)
        dis=disparityregression(self.maxdisp)(x)
        return dis



