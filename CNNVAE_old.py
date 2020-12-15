import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import jit

def createBasicConv(in_chs, out_chs, **kwargs):
    return nn.Sequential(
        nn.Conv1d(in_chs, out_chs, **kwargs),
        nn.ReLU(inplace=True)
        )

def createNormConv(in_chs, out_chs, **kwargs):
    return nn.Sequential(
        nn.Conv1d(in_chs, out_chs, **kwargs),
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

class Encoder(nn.Module):
# class Encoder(jit.ScriptModule):

    def __init__(self, first_channel, latent_size, batchNorm=False):
        super().__init__()

        if batchNorm:
            createConv = createBasicConv
        else:
            createConv = createNormConv

        k = first_channel

        self.conv1 = createConv(   1,    k, kernel_size=3, stride=3, padding=1)
        self.conv2 = createConv(   k,  2*k, kernel_size=3, stride=3, padding=1)
        self.conv3 = createConv( 2*k,  4*k, kernel_size=3, stride=3, padding=1)
        self.conv4 = createConv( 4*k,  8*k, kernel_size=3, stride=2, padding=1)
        self.conv5 = createConv( 8*k, 16*k, kernel_size=3, stride=2, padding=1)
        self.conv6 = createConv(16*k, 32*k, kernel_size=3, stride=2, padding=1)
        
        # self.conv1 = createConv(   1,    k, kernel_size=3, stride=2, padding=1)
        # self.conv2 = createConv(   k,  2*k, kernel_size=3, stride=2, padding=1)
        # self.conv3 = createConv( 2*k,  4*k, kernel_size=3, stride=2, padding=1)
        # self.conv4 = createConv( 4*k,  8*k, kernel_size=3, stride=3, padding=1)
        # self.conv5 = createConv( 8*k, 16*k, kernel_size=3, stride=3, padding=1)
        # self.conv6 = createConv(16*k, 32*k, kernel_size=3, stride=3, padding=1)

        self.embedding_size = 5*32*k

        self.fc1 = nn.Linear(self.embedding_size, latent_size)
        self.fc2 = nn.Linear(self.embedding_size, latent_size)

    def forward(self, x):

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


class Decoder(nn.Module):
# class Decoder(jit.ScriptModule):

    def __init__(self, last_channel, latent_size, batchNorm=False):
        super().__init__()

        if batchNorm:
            createConvT = createBasicConvT
            createLastConvT = createBasicLastConvT
        else:
            createConvT = createNormConvT
            createLastConvT = createBasicLastConvT

        k = last_channel

        self.embedding_size = 5*32*k

        self.fc = nn.Linear(latent_size, self.embedding_size)

        self.convT1 = createConvT(32*k, 16*k, kernel_size=2, stride=2, padding=0)
        self.convT2 = createConvT(16*k,  8*k, kernel_size=2, stride=2, padding=0)
        self.convT3 = createConvT( 8*k,  4*k, kernel_size=2, stride=2, padding=0)
        self.convT4 = createConvT( 4*k,  2*k, kernel_size=3, stride=3, padding=0)
        self.convT5 = createConvT( 2*k,    k, kernel_size=3, stride=3, padding=0)
        self.convT6 = createLastConvT(   k,    1, kernel_size=3, stride=3, padding=0)

        # self.convT1 = createConvT(32*k, 16*k, kernel_size=3, stride=3, padding=0)
        # self.convT2 = createConvT(16*k,  8*k, kernel_size=3, stride=3, padding=0)
        # self.convT3 = createConvT( 8*k,  4*k, kernel_size=3, stride=3, padding=0)
        # self.convT4 = createConvT( 4*k,  2*k, kernel_size=2, stride=2, padding=0)
        # self.convT5 = createConvT( 2*k,    k, kernel_size=2, stride=2, padding=0)
        # self.convT6 = createLastConvT(   k,    1, kernel_size=2, stride=2, padding=0)

        # self.convT1 = createConvT(32*k, 16*k, kernel_size=4, stride=2, padding=1)
        # self.convT2 = createConvT(16*k,  8*k, kernel_size=4, stride=2, padding=1)
        # self.convT3 = createConvT( 8*k,  4*k, kernel_size=4, stride=2, padding=1)
        # self.convT4 = createConvT( 4*k,  2*k, kernel_size=5, stride=3, padding=1)
        # self.convT5 = createConvT( 2*k,    k, kernel_size=5, stride=3, padding=1)
        # self.convT6 = createLastConvT(   k,    1, kernel_size=5, stride=3, padding=1)


    def forward(self, z):

        x = self.fc(z)

        x = x.view(-1, self.embedding_size//5, 5)

        x = self.convT1(x)
        x = self.convT2(x)
        x = self.convT3(x)
        x = self.convT4(x)
        x = self.convT5(x)
        x = self.convT6(x)

        return x

# class CNNVAE(nn.Module):
class CNNVAE(jit.ScriptModule):

    def __init__(self, first_channel, latent_size, batchNorm=False):
        super().__init__()
        self.encoder = Encoder(first_channel, latent_size, batchNorm)
        self.decoder = Decoder(first_channel, latent_size, batchNorm)

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

def testEncode():
    x = torch.randn(10,1,1080)
    print(x.size())

    print("stride 2")

    conv3_s2 = nn.Conv1d(1, 1, kernel_size=3, stride=2, padding=1)
    print(conv3_s2(x).size())

    conv5_s2 = nn.Conv1d(1, 1, kernel_size=5, stride=2, padding=2)
    print(conv5_s2(x).size())

    conv7_s2 = nn.Conv1d(1, 1, kernel_size=7, stride=2, padding=3)
    print(conv7_s2(x).size())

    print("stride 3")

    conv3_s3 = nn.Conv1d(1, 1, kernel_size=3, stride=3, padding=1)
    print(conv3_s3(x).size())

    conv5_s3 = nn.Conv1d(1, 1, kernel_size=5, stride=3, padding=2)
    print(conv5_s3(x).size())

    conv7_s3 = nn.Conv1d(1, 1, kernel_size=7, stride=3, padding=3)
    print(conv7_s3(x).size())

    print("stride 5")

    conv3_s5 = nn.Conv1d(1, 1, kernel_size=3, stride=5, padding=1)
    print(conv3_s5(x).size())

    conv5_s5 = nn.Conv1d(1, 1, kernel_size=5, stride=5, padding=2)
    print(conv5_s5(x).size())

    conv7_s5 = nn.Conv1d(1, 1, kernel_size=7, stride=5, padding=3)
    print(conv7_s5(x).size())

    print("Encode")

    y1 = x
    
    y1 = conv3_s3(y1)
    print(y1.size())
    y1 = conv3_s3(y1)
    print(y1.size())
    y1 = conv3_s3(y1)
    print(y1.size())

    y1 = conv3_s2(y1)
    print(y1.size())
    y1 = conv3_s2(y1)
    print(y1.size())
    y1 = conv3_s2(y1)
    print(y1.size())

def testDecode():
    x = torch.randn(10,1,5)
    print(x.size())

    print("stride 3")

    conv3_s2  = nn.ConvTranspose1d(1, 1, kernel_size=2, stride=2, padding=0)
    conv3_s3  = nn.ConvTranspose1d(1, 1, kernel_size=3, stride=3, padding=0)

    y1 = x

    y1 = conv3_s2(y1)
    print(y1.size())
    y1 = conv3_s2(y1)
    print(y1.size())
    y1 = conv3_s2(y1)
    print(y1.size())

    y1 = conv3_s3(y1)
    print(y1.size())
    y1 = conv3_s3(y1)
    print(y1.size())
    y1 = conv3_s3(y1)
    print(y1.size())

if __name__ == '__main__':
    # print("testEncode")
    # testEncode()
    # print("testDecode")
    # testDecode()


    encoder = Encoder(first_channel=8, latent_size=18)

    x = torch.randn(10,1,1080)
    mu, sigma = encoder(x)

    print(mu.size())

    decoder = Decoder(last_channel=8, latent_size=18)

    recon = decoder(mu)

    print(recon.size())
