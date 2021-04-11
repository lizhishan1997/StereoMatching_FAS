import torch
import torch.nn as nn

class ClassificationNet(torch.nn.Module):
    def __init__(self):
        super(ClassificationNet, self).__init__()
        self.feature=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=16,padding=1,kernel_size=3,stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, padding=1, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, padding=1, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, padding=1, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, padding=1, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d((4, 4)),
        )
        self.fc=nn.Linear(128*2*2,2)

    def forward(self, x):
        x=self.feature(x)
        x=x.view(x.size(0),-1)
        x=self.fc(x)
        return x