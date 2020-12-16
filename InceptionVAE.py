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


def createBasicLastConvT(in_chs, out_chs, **kwargs):
    return nn.Sequential(
        nn.ConvTranspose1d(in_chs, out_chs, **kwargs),
        nn.Sigmoid()
        )

def createNormLastConvT(in_chs, out_chs, **kwargs):
    return nn.Sequential(
        nn.ConvTranspose1d(in_chs, out_chs, **kwargs),
        nn.BatchNorm1d(out_chs),
        nn.Sigmoid()
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


# class Encoder(nn.Module):
class Encoder(jit.ScriptModule):

    def __init__(self, first_channel, latent_size, repeat=0, batchNorm=False):
        super().__init__()

        if batchNorm:
            createConv = createNormConv
        else:
            createConv = createBasicConv

        k = first_channel

        self.repeat = repeat

        self.conv1 = createConv(   1,    k, kernel_size=3, stride=3, padding=1)
        self.conv2 = createConv(   k,  2*k, kernel_size=3, stride=3, padding=1)
        self.conv3 = createConv( 2*k,  4*k, kernel_size=3, stride=3, padding=1)
        self.conv4 = createConv( 4*k,  8*k, kernel_size=3, stride=2, padding=1)
        self.conv5 = createConv( 8*k, 16*k, kernel_size=3, stride=2, padding=1)
        self.conv6 = createConv(16*k, 32*k, kernel_size=3, stride=2, padding=1)
        
        # self.module1 = nn.Sequential(*[EncoderInceptionBasic(   k, batchNorm) for _ in range(self.repeat)])
        # self.module2 = nn.Sequential(*[EncoderInceptionBasic( 2*k, batchNorm) for _ in range(self.repeat)])
        # self.module3 = nn.Sequential(*[EncoderInceptionBasic( 4*k, batchNorm) for _ in range(self.repeat)])
        # self.module4 = nn.Sequential(*[EncoderInceptionBasic( 8*k, batchNorm) for _ in range(self.repeat)])
        # self.module5 = nn.Sequential(*[EncoderInceptionBasic(16*k, batchNorm) for _ in range(self.repeat)])
        # self.module6 = nn.Sequential(*[EncoderInceptionBasic(32*k, batchNorm) for _ in range(self.repeat)])

        # self.module0 = nn.Sequential(*[EncoderInceptionBasic(   1) for _ in range(self.repeat)])

        self.module1 = nn.Sequential(*[EncoderInceptionBasic(   k) for _ in range(self.repeat)])
        self.module2 = nn.Sequential(*[EncoderInceptionBasic( 2*k) for _ in range(self.repeat)])
        self.module3 = nn.Sequential(*[EncoderInceptionBasic( 4*k) for _ in range(self.repeat)])
        self.module4 = nn.Sequential(*[EncoderInceptionBasic( 8*k) for _ in range(self.repeat)])
        self.module5 = nn.Sequential(*[EncoderInceptionBasic(16*k) for _ in range(self.repeat)])
        self.module6 = nn.Sequential(*[EncoderInceptionBasic(32*k) for _ in range(self.repeat)])

        self.embedding_size = 5*32*k

        self.fc1 = nn.Linear(self.embedding_size, latent_size)
        self.fc2 = nn.Linear(self.embedding_size, latent_size)

    def forward(self, x):

        if self.repeat > 0:
            # h = self.module0(x)

            # h = self.conv1(h)
            h = self.conv1(x)
            h = self.module1(h)
            h = self.conv2(h)
            h = self.module2(h)
            h = self.conv3(h)
            h = self.module3(h)
            h = self.conv4(h)
            h = self.module4(h)
            h = self.conv5(h)
            h = self.module5(h)
            h = self.conv6(h)
            h = self.module6(h)
        else:
            h = self.conv1(x)
            h = self.conv2(h)
            h = self.conv3(h)
            h = self.conv4(h)
            h = self.conv5(h)
            h = self.conv6(h)

        h = h.view(-1, self.embedding_size)

        mu     = self.fc1(h)
        logvar = self.fc2(h)

        return mu, logvar

# class Decoder(nn.Module):
class Decoder(jit.ScriptModule):

    def __init__(self, last_channel, latent_size, repeat=0, batchNorm=False):
        super().__init__()

        if batchNorm:
            createConvT = createNormConvT
            createLastConvT = createBasicLastConvT
        else:
            createConvT = createBasicConvT
            createLastConvT = createBasicLastConvT

        self.repeat = repeat

        k = last_channel

        self.embedding_size = 5*32*k

        self.fc = nn.Linear(latent_size, self.embedding_size)

        self.convT1 =     createConvT(32*k, 16*k, kernel_size=2, stride=2, padding=0)
        self.convT2 =     createConvT(16*k,  8*k, kernel_size=2, stride=2, padding=0)
        self.convT3 =     createConvT( 8*k,  4*k, kernel_size=2, stride=2, padding=0)
        self.convT4 =     createConvT( 4*k,  2*k, kernel_size=3, stride=3, padding=0)
        self.convT5 =     createConvT( 2*k,    k, kernel_size=3, stride=3, padding=0)
        # self.convT6 =     createConvT(   k,    1, kernel_size=3, stride=3, padding=0)
        self.convT6 = createLastConvT(   k,    1, kernel_size=3, stride=3, padding=0)

        # self.module1 = nn.Sequential(*[DecoderInceptionBasic(32*k, batchNorm) for _ in range(self.repeat)])
        # self.module2 = nn.Sequential(*[DecoderInceptionBasic(16*k, batchNorm) for _ in range(self.repeat)])
        # self.module3 = nn.Sequential(*[DecoderInceptionBasic( 8*k, batchNorm) for _ in range(self.repeat)])
        # self.module4 = nn.Sequential(*[DecoderInceptionBasic( 4*k, batchNorm) for _ in range(self.repeat)])
        # self.module5 = nn.Sequential(*[DecoderInceptionBasic( 2*k, batchNorm) for _ in range(self.repeat)])
        # self.module6 = nn.Sequential(*[DecoderInceptionBasic(   k, batchNorm) for _ in range(self.repeat)])

        self.module1 = nn.Sequential(*[DecoderInceptionBasic(32*k) for _ in range(self.repeat)])
        self.module2 = nn.Sequential(*[DecoderInceptionBasic(16*k) for _ in range(self.repeat)])
        self.module3 = nn.Sequential(*[DecoderInceptionBasic( 8*k) for _ in range(self.repeat)])
        self.module4 = nn.Sequential(*[DecoderInceptionBasic( 4*k) for _ in range(self.repeat)])
        self.module5 = nn.Sequential(*[DecoderInceptionBasic( 2*k) for _ in range(self.repeat)])
        self.module6 = nn.Sequential(*[DecoderInceptionBasic(   k) for _ in range(self.repeat)])

        # self.module7 = nn.Sequential(*[DecoderInceptionBasic(   1) for _ in range(self.repeat)])


    def forward(self, z):

        x = self.fc(z)

        x = x.view(-1, self.embedding_size//5, 5)

        if self.repeat > 0:
            x = self.module1(x)
            x = self.convT1(x)
            x = self.module2(x)
            x = self.convT2(x)
            x = self.module3(x)
            x = self.convT3(x)
            x = self.module4(x)
            x = self.convT4(x)
            x = self.module5(x)
            x = self.convT5(x)
            x = self.module6(x)
            x = self.convT6(x)

            # x = self.module7(x)
            # x = F.sigmoid(x)

        else:
            x = self.convT1(x)
            x = self.convT2(x)
            x = self.convT3(x)
            x = self.convT4(x)
            x = self.convT5(x)
            x = self.convT6(x)

        return x

# class InceptionVAE(nn.Module):
class InceptionVAE(jit.ScriptModule):

    def __init__(self, first_channel, latent_size, repeat=0, batchNorm=False):
        super().__init__()
        self.encoder = Encoder(first_channel, latent_size, repeat, batchNorm)
        self.decoder = Decoder(first_channel, latent_size, repeat, batchNorm)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        sigma = logvar.mul(0.5).exp()
        eps   = torch.randn_like(sigma)

        if self.training:
            z = eps.mul(sigma).add_(mu) # mu + sigma^(1/2) * eps
        else:
            z = mu

        recon_x = self.decoder(z)
        return recon_x, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):

        assert (torch.min(recon_x) >= 0. and torch.max(recon_x) <= 1.)

        x2 = torch.clamp(x.view(recon_x.size()), 1e-24, 1.0-1e-24)
        assert (torch.min(x2) >= 0. and torch.max(x2) <= 1.)

        BCE = F.binary_cross_entropy(recon_x, x2, reduction='sum')
        # BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.x_size), reduction='sum')
        # BCE = F.mse_loss(recon_x, x)
 
        # 0.5*(1 + log(sigma^2) - mu^2 - sigma^2) 
        # 実装ではsigmaがマイナスになるとlogsigmaを求められないためか、2*logsigmaをlogvarと置いて
        # KL距離を0.5*(mu^2 + exp(logvar) −logvar − 1) とする記述が主流?
        # https://qiita.com/nishiha/items/2264da933504fbe3fc68

        KLD = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)
        # KLD = 0

        return BCE, KLD

if __name__ == '__main__':

    x = torch.randn(10,1,1080)

    model = EncoderInceptionBasic(in_channels=1)

    out = model(x)

    print(out.size())

    model = DecoderInceptionBasic(in_channels=1)

    out = model(x)

    print(out.size())

    encoder = Encoder(first_channel=8, latent_size=18, repeat=1, batchNorm=False)

    mu, sigma = encoder(x)

    print(mu.size())

    decoder = Decoder(last_channel=8, latent_size=18, repeat=1, batchNorm=False)

    recon = decoder(mu)

    print(recon.size())