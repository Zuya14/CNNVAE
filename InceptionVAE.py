import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import jit

def createBasicConv(in_chs, out_chs, **kwargs):
    return nn.Sequential(
        nn.Conv1d(in_chs, out_chs, bias=False, **kwargs),
        nn.ReLU(inplace=True)
        )

def createNormConv(in_chs, out_chs, **kwargs):
    return nn.Sequential(
        nn.Conv1d(in_chs, out_chs, bias=False, **kwargs),
        nn.BatchNorm1d(out_chs),
        nn.ReLU(inplace=True)
        )


def createBasicConvT(in_chs, out_chs, **kwargs):
    return nn.Sequential(
        nn.ConvTranspose1d(in_chs, out_chs, **kwargs),
        nn.ReLU(inplace=True)
        )

def createNormConvT(in_chs, out_chs, **kwargs):
    return nn.Sequential(
        nn.ConvTranspose1d(in_chs, out_chs, **kwargs),
        nn.BatchNorm1d(out_chs),
        nn.ReLU(inplace=True)
        )


class EncoderInceptionBasic(nn.Module):

    def __init__(self, in_channels, batchNorm=False):
        super().__init__()

        if batchNorm:
            createConv = createBasicConv
        else:
            createConv = createNormConv

        self.conv1 = createConv(in_channels, in_channels, kernel_size=1)

        self.conv3 = createConv(in_channels, in_channels, kernel_size=3, padding=1)

        self.conv3_1 = createConv(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv3_2 = createConv(in_channels, in_channels, kernel_size=3, padding=1)

        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        
        y_c1 = self.conv1(x)

        y_c3 = self.conv3(x)

        y_c5 = self.conv3_1(x)
        y_c5 = self.conv3_2(y_c5)
        
        y_p3 = self.pool3(x)

        # print(y_c1.size())
        # print(y_c3.size())
        # print(y_c5.size())
        # print(y_p3.size())

        out = y_c1 + y_c3 + y_c5 + y_p3

        return out

class DecoderInceptionBasic(nn.Module):

    def __init__(self, in_channels, batchNorm=False):
        super().__init__()

        if batchNorm:
            createConv = createBasicConvT
        else:
            createConv = createNormConvT

        self.conv1 = createConv(in_channels, in_channels, kernel_size=1)

        self.conv3 = createConv(in_channels, in_channels, kernel_size=3, padding=1)

        self.conv3_1 = createConv(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv3_2 = createConv(in_channels, in_channels, kernel_size=3, padding=1)

        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        
        y_c1 = self.conv1(x)

        y_c3 = self.conv3(x)

        y_c5 = self.conv3_1(x)
        y_c5 = self.conv3_2(y_c5)
        
        y_p3 = self.pool3(x)

        # print(y_c1.size())
        # print(y_c3.size())
        # print(y_c5.size())
        # print(y_p3.size())

        out = y_c1 + y_c3 + y_c5 + y_p3

        return out


class Encoder(nn.Module):

    def __init__(self, latent_size, cnn_outsize):
        super().__init__()

        assert len(channels) >= 2

        self.cnn_outsize = cnn_outsize

        layer = []
        for i in range(len(channels)-1):
            layer.append(createEncoderUnit(channels[i], channels[i+1]))
            # layer.extend(createEncoderUnit(channels[i], channels[i+1]).children())
        
        self.layers = nn.ModuleList(layer)

        self.fc1 = nn.Linear(cnn_outsize, latent_size)
        self.fc2 = nn.Linear(cnn_outsize, latent_size)

if __name__ == '__main__':

    x = torch.randn(10,1,1080)

    model = EncoderInceptionBasic(in_channels=1)

    out = model(x)

    print(out.size())

    model = DecoderInceptionBasic(in_channels=1)

    out = model(x)

    print(out.size())
