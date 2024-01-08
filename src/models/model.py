import torch
from torch import nn
import torch.nn.functional as F

class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3*224*224, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 10)
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.log_softmax(self.fc4(x), dim=1)
        
        return x

class MyCNNModel(nn.Module):
    def __init__(self):
        super(MyCNNModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 112 * 112, 256)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(256, 5)  # Assuming 5 classes for classification

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, num_classes=5):
        super(UNet, self).__init__()

        # Contracting Path (Encoder)
        self.enc_conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc_conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Middle Part
        self.middle_conv1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.middle_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # Expanding Path (Decoder)
        self.up_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.dec_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.up_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dec_conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Final Classifier
        self.final_fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # Encoder
        e1 = F.relu(self.enc_conv1(x))
        e1 = F.relu(self.enc_conv2(e1))
        e1_pool = self.pool1(e1)

        e2 = F.relu(self.enc_conv3(e1_pool))
        e2 = F.relu(self.enc_conv4(e2))
        e2_pool = self.pool2(e2)

        # Middle
        m = F.relu(self.middle_conv1(e2_pool))
        m = F.relu(self.middle_conv2(m))

        # Decoder
        d1 = self.up_conv1(m)
        d1 = torch.cat((e2, d1), dim=1)
        d1 = F.relu(self.dec_conv1(d1))
        d1 = F.relu(self.dec_conv2(d1))

        d2 = self.up_conv2(d1)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = F.relu(self.dec_conv3(d2))
        d2 = F.relu(self.dec_conv4(d2))

        # Global Average Pooling
        pooled = self.global_avg_pool(d2)
        pooled = pooled.view(pooled.size(0), -1)  # Flatten the output

        # Classifier
        out = self.final_fc(pooled)
        return out
