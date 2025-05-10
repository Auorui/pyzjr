"""
Copyright (c) 2024, Auorui.
All rights reserved.
http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf
"""
import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=4)
        self.fc1 = nn.Linear(120, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.sigmoid(x)
        x = self.max_pool1(x)
        x = torch.sigmoid(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        # 输入层将数据拉平[B,C,H,W] -> [B,C*H*W]
        x = torch.reshape(x, [x.shape[0], -1])
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        return x


if __name__=='__main__':
    import torchsummary
    model = LeNet()
    print(model)
    torchsummary.summary(model, (1, 28, 28))
    # Total params: 41,806


"""LeNet
--------------------------------------------
Layer (type)               Output Shape
============================================
Conv2d-1                  [-1, 6, 24, 24]    
MaxPool2d-2               [-1, 6, 12, 12]          
Conv2d-3                  [-1, 16, 8, 8]         
MaxPool2d-4               [-1, 16, 4, 4]              
Conv2d-5                  [-1, 120, 1, 1]       
Linear-6                  [-1, 64]        
Linear-7                  [-1, 10]             
============================================
"""