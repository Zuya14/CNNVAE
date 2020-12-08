import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch import jit

def createEncoderUnit(in_chs, out_chs):
    return nn.Sequential(
        nn.Conv1d(in_chs, out_chs, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv1d(out_chs, out_chs, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(kernel_size=2, stride=2)
        )

def createDecoderUnit(in_chs, out_chs):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ConvTranspose1d(in_chs, out_chs, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose1d(out_chs, out_chs, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True)
        )

def createDecoderUnitLast(in_chs, out_chs):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ConvTranspose1d(in_chs, out_chs, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose1d(out_chs, out_chs, kernel_size=3, stride=1, padding=1)
        )

class Encoder(nn.Module):

    def __init__(self, channels, latent_size, cnn_outsize):
        super().__init__()

        assert len(channels) >= 2

        self.cnn_outsize = cnn_outsize

        layer = []
        for i in range(len(channels)-1):
            layer.append(createEncoderUnit(channels[i], channels[i+1]))
        
        self.layers = nn.ModuleList(layer)

        self.fc1 = nn.Linear(cnn_outsize, latent_size)
        self.fc2 = nn.Linear(cnn_outsize, latent_size)

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = layer(h)
            # print(h.size())
        h = h.view(-1, self.cnn_outsize)

        mu     = self.fc1(h)
        logvar = self.fc2(h)
        # print(mu.size())

        return mu, logvar

class Decoder(nn.Module):

    def __init__(self, channels, latent_size, cnn_outsize):
        super().__init__()

        assert len(channels) >= 2

        self.cnn_outsize = cnn_outsize

        self.fc = nn.Linear(latent_size, cnn_outsize)

        layer = []
        for i in range(len(channels)-2):
            layer.append(createDecoderUnit(channels[i], channels[i+1]))

        layer.append(createDecoderUnit(channels[-2], channels[-1]))

        self.layers = nn.ModuleList(layer)

        self.last_channel = channels[0]

    def forward(self, z):
        x = self.fc(z)

        x = x.view(-1, self.last_channel, int(self.cnn_outsize/self.last_channel))
        # print(x.size())

        for layer in self.layers:
            x = layer(x)
            # print(x.size())

        return x

if __name__ == '__main__':

    vgg = createEncoderUnit(1, 64)
    vgg2 = createEncoderUnit(64, 128)
    vgg3 = createEncoderUnit(128, 256)

    dvgg3 = createDecoderUnit(256, 128)
    dvgg2 = createDecoderUnit(128, 64)
    dvgg = createDecoderUnit(64, 1)

    x = torch.randn(10,1,1080)
    print(x.size())

    # y = vgg(x)
    # print(y.size())
    
    # z = vgg2(y)
    # print(z.size())

    # a = vgg3(z)
    # print(a.size())

    # dz = dvgg3(a)
    # print(dz.size())

    # dy = dvgg2(dz)
    # print(dy.size())

    # dx = dvgg(dy)
    # print(dx.size())

    latent = 18

    encoder = Encoder(channels=[1, 64, 128, 256], latent_size=latent, cnn_outsize=34560)
    m, s = encoder(x)
    print(m.size())

    decoder = Decoder(channels=list(reversed([1, 64, 128, 256])), latent_size=latent, cnn_outsize=34560)
    recon = decoder(m)
    print(recon.size())