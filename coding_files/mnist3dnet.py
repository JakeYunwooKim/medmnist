# This file contains definitions of three networks for deep learning MedMNIST 3D data.

import torch.nn as nn
import inspect

class Net1(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Net1, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = x.float()
        x = self.layer1(x)
        x = self.layer2(x)
        x = nn.functional.adaptive_avg_pool3d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def __str__(self):
        return self.__class__.__name__ + "\n" + inspect.getsource(self.__init__) + "\n" + inspect.getsource(self.forward)


class Net2(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Net2, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = x.float()
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = nn.functional.adaptive_avg_pool3d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def __str__(self):
        return self.__class__.__name__ + "\n" + inspect.getsource(self.__init__) + "\n" + inspect.getsource(self.forward)


class Net3(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Net3, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = x.float()
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = nn.functional.adaptive_avg_pool3d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def __str__(self):
        return self.__class__.__name__ + "\n" + inspect.getsource(self.__init__) + "\n" + inspect.getsource(self.forward)
