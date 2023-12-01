import torch
import torch.nn as nn
import torch as T
import cv2
import pickle
from torch import optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from lstm import CLSTM, CLSTM_cell

def double_convolution(in_channels, out_channels):
    """
    In the original paper implementation, the convolution operations were
    not padded but we are padding them here. This is because, we need the
    output result size to be same as input size.
    """
    conv_op = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )
    return conv_op


class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()

        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

        num_features = 1024
        filter_size = 5
        batch_size = 10
        shape = (8, 16)  # H,W
        inp_chans = 1024
        nlayers = 2
        seq_len = 5

        # If using this format, then we need to transpose in CLSTM
        self.conv_lstm = CLSTM(shape, inp_chans, filter_size, num_features, nlayers)
        self.conv_lstm.apply(weights_init)
        self.conv_lstm.cuda()

        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        # Contracting path.
        # Each convolution is applied twice.
        self.down_convolution_1 = double_convolution(1, 64)
        self.down_convolution_2 = double_convolution(64, 128)
        self.down_convolution_3 = double_convolution(128, 256)
        self.down_convolution_4 = double_convolution(256, 512)
        self.down_convolution_5 = double_convolution(512, 1024)
        # Expanding path.
        self.up_transpose_1 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512,
            kernel_size=2,
            stride=2)
        # Below, `in_channels` again becomes 1024 as we are concatinating.
        self.up_convolution_1 = double_convolution(1024, 512)
        self.up_transpose_2 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256,
            kernel_size=2,
            stride=2)
        self.up_convolution_2 = double_convolution(512, 256)
        self.up_transpose_3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128,
            kernel_size=2,
            stride=2)
        self.up_convolution_3 = double_convolution(256, 128)
        self.up_transpose_4 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64,
            kernel_size=2,
            stride=2)
        self.up_convolution_4 = double_convolution(128, 64)
        # output => `out_channels` as per the number of classes.
        self.out = nn.Conv2d(
            in_channels=64, out_channels=num_classes,
            kernel_size=1
        )

    def forward(self, x):
        x = torch.unbind(x, dim=1)
        data = []
        # print(len(x))
        for item in x:
            # print(item.shape)
            down_1 = self.down_convolution_1(item)
            down_2 = self.max_pool2d(down_1)
            down_3 = self.down_convolution_2(down_2)
            down_4 = self.max_pool2d(down_3)
            down_5 = self.down_convolution_3(down_4)
            down_6 = self.max_pool2d(down_5)
            down_7 = self.down_convolution_4(down_6)
            down_8 = self.max_pool2d(down_7)
            down_9 = self.down_convolution_5(down_8)
            data.append(down_9.unsqueeze(0))

        # *** DO NOT APPLY MAX POOL TO down_9 ***
        data = torch.cat(data, dim=0)
        data = torch.permute(data, (1, 0, 2, 3, 4))
        hidden_state = self.conv_lstm.init_hidden(8)
        # print("data shape", data.shape)
        lstm = self.conv_lstm(data, hidden_state)[1]
        test = lstm[-1]
        # print("test shape: ", test.shape)


        up_1 = self.up_transpose_1(test)
        # print(up_1.shape)
        # print(down_7.shape)
        x = self.up_convolution_1(torch.cat([down_7, up_1], 1))
        up_2 = self.up_transpose_2(x)
        x = self.up_convolution_2(torch.cat([down_5, up_2], 1))
        up_3 = self.up_transpose_3(x)
        x = self.up_convolution_3(torch.cat([down_3, up_3], 1))
        up_4 = self.up_transpose_4(x)
        x = self.up_convolution_4(torch.cat([down_1, up_4], 1))
        out = self.out(x)
        # return F.softmax(out, 1)
        return out