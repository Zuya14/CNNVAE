import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import jit

def createEncoderUnit(in_chs, out_chs):
    return nn.Sequential(
        nn.Conv1d(in_chs, out_chs, kernel_size=3, padding=1),
        # nn.BatchNorm1d(out_chs).cuda(),
        nn.ReLU(inplace=True),
        nn.Conv1d(out_chs, out_chs, kernel_size=3, padding=1),
        # nn.BatchNorm1d(out_chs).cuda(),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(kernel_size=2, stride=2)
        )

def createDecoderUnit(in_chs, out_chs):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ConvTranspose1d(in_chs, out_chs, kernel_size=3, stride=1, padding=1),
        # nn.BatchNorm1d(out_chs).cuda(),
        nn.ReLU(inplace=True),
        nn.ConvTranspose1d(out_chs, out_chs, kernel_size=3, stride=1, padding=1),
        # nn.BatchNorm1d(out_chs).cuda(),
        nn.ReLU(inplace=True)
        )

# def createDecoderUnitLast(in_chs, out_chs):
#     return nn.Sequential(
#         nn.Upsample(scale_factor=2, mode='nearest'),
#         nn.ConvTranspose1d(in_chs, out_chs, kernel_size=3, stride=1, padding=1),
#         nn.ReLU(inplace=True),
#         nn.ConvTranspose1d(out_chs, out_chs, kernel_size=3, stride=1, padding=1)
#         )

def createDecoderUnitLast(in_chs, out_chs):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ConvTranspose1d(in_chs, out_chs, kernel_size=3, stride=1, padding=1),
        # nn.BatchNorm1d(out_chs).cuda(),
        nn.ReLU(inplace=True),
        nn.ConvTranspose1d(out_chs, out_chs, kernel_size=3, stride=1, padding=1),
        nn.Sigmoid()
        )



# def createDecoderUnit(in_chs, out_chs):
#     return nn.Sequential(
#         nn.ConvTranspose1d(in_chs, out_chs, kernel_size=3, stride=2, padding=1),
#         nn.ReLU(inplace=True),
#         nn.ConvTranspose1d(out_chs, out_chs, kernel_size=3, stride=1, padding=1),
#         nn.ReLU(inplace=True)
#         )

# def createDecoderUnitLast(in_chs, out_chs):
#     return nn.Sequential(
#         nn.ConvTranspose1d(in_chs, out_chs, kernel_size=3, stride=2, padding=1),
#         nn.ReLU(inplace=True),
#         nn.ConvTranspose1d(out_chs, out_chs, kernel_size=3, stride=1, padding=1)
#         )

# class Encoder(nn.Module):
class Encoder(jit.ScriptModule):

    def __init__(self, channels, latent_size, cnn_outsize):
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

# class Decoder(nn.Module):
class Decoder(jit.ScriptModule):

    def __init__(self, channels, latent_size, cnn_outsize):
        super().__init__()

        assert len(channels) >= 2

        self.cnn_outsize = cnn_outsize

        self.fc = nn.Linear(latent_size, cnn_outsize)

        layer = []
        for i in range(len(channels)-2):
            layer.append(createDecoderUnit(channels[i], channels[i+1]))
            # layer.extend(createDecoderUnit(channels[i], channels[i+1]).children())

        layer.append(createDecoderUnitLast(channels[-2], channels[-1]))
        # layer.extend(createDecoderUnitLast(channels[-2], channels[-1]).children())

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

# class VGGVAE(nn.Module):
class VGGVAE(jit.ScriptModule):

    def __init__(self, channels, latent_size, cnn_outsize):
        super().__init__()
        # self.x_size  = x_size
        self.encoder = Encoder(channels, latent_size, cnn_outsize)
        self.decoder = Decoder(list(reversed(channels)), latent_size, cnn_outsize)

    def forward(self, x):
        # mu, logvar = self.encoder(x.view(-1, self.x_size))
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

        BCE = F.binary_cross_entropy(recon_x, torch.clamp(x.view(recon_x.size()), 1e-24, 1.0-1e-24), reduction='sum')
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

    # vgg = createEncoderUnit(1, 64)
    # vgg2 = createEncoderUnit(64, 128)
    # vgg3 = createEncoderUnit(128, 256)

    # dvgg3 = createDecoderUnit(256, 128)
    # dvgg2 = createDecoderUnit(128, 64)
    # dvgg = createDecoderUnit(64, 1)

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
    outsize = 34560

    encoder = Encoder(channels=[1, 64, 128, 256], latent_size=latent, cnn_outsize=outsize)
    m, s = encoder(x)
    print(m.size())

    decoder = Decoder(channels=list(reversed([1, 64, 128, 256])), latent_size=latent, cnn_outsize=outsize)
    recon = decoder(m)
    print(recon.size())

    vae = VGGVAE(channels=[1, 64, 128, 256], latent_size=latent, cnn_outsize=outsize)
    recon_x, mu, logvar = vae(x)
    print(recon_x.size())
