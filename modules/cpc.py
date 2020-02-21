import torch
from .module import Module


class CPC(torch.nn.Module):
    def __init__(
        self, args, strides, filter_sizes, padding, genc_hidden, gar_hidden,
    ):
        super(CPC, self).__init__()

        self.args = args
        self.strides = strides
        self.filter_sizes = filter_sizes
        self.padding = padding
        self.genc_input = 1
        self.genc_hidden = genc_hidden
        self.gar_hidden = gar_hidden

        self.module = Module(
            args, strides, filter_sizes, padding, self.genc_input, genc_hidden, gar_hidden
        )

        # initialize module list
        # self.model = torch.nn.ModuleList([self.module])

    def forward(self, x):
        """Forward through the network"""
        # loss = torch.zeros(len(self.model))
        # accuracy = torch.zeros(len(self.model))

        # for idx, module in enumerate(self.model):
        #     loss[idx], accuracy[idx], _, z = module(x)
        #     x = z.permute(0, 2, 1).detach()

        loss, accuracy, _, z = self.module(x)
        # x = z.permute(0, 2, 1)
        return loss
