import argparse
import torch
from VGGVAE import Encoder


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='cnnout')

    parser.add_argument("--channels", nargs="*", type=int, default=[1, 64])
    parser.add_argument("--latent", type=int, default=18)

    args = parser.parse_args()


    encoder = Encoder(channels=args.channels, latent_size=args.latent, cnn_outsize=1)

    h = torch.randn(1,1,1080)

    for layer in encoder .layers:
        h = layer(h)

    print(h.size())
    print("outsize:", h.size()[1]*h.size()[2])