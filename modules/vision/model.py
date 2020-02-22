import torch

from .resnet_encoder import ResNetEncoder, PreActBottleneckNoBN
from .autoregressor import Autoregressor


class Model(torch.nn.Module):
    def __init__(self, args, block_dims, num_channels):
        super(Model, self).__init__()

        self.args = args

        if args.resnet == 34:
            self.block = PreActBlockNoBN
        elif args.resnet == 50:
            self.block = PreActBottleneckNoBN
        else:
            raise Exception("Illegal choice of |args.resnet|")

        if args.grayscale:
            input_dim = 1
        else:
            input_dim = 3

        output_dim = num_channels[-1] * self.block.expansion
        self.encoder = ResNetEncoder(
            args,
            self.block,
            block_dims,
            num_channels,
            0,
            calc_loss=False,
            input_dim=input_dim,
        )

        self.autoregressor = Autoregressor(args, in_channels=output_dim, calc_loss=True)

        self.model = torch.nn.ModuleList([self.encoder, self.autoregressor])

    def forward(self, x, label):
        """Forward through the network"""

        n_patches_x, n_patches_y = None, None
        h, z, loss, accuracy, n_patches_x, n_patches_y = self.encoder(
            x, n_patches_x, n_patches_y, label
        )

        c, loss = self.autoregressor(h)
        return loss, accuracy, c, h
