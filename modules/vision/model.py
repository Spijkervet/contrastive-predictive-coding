import torch

# from .cpc import CPC


class Model(torch.nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.args = args

        self.model = torch.nn.Linear(1,1)
        # self.model = CPC(
        #     args,
        #     strides,
        #     filter_sizes,
        #     padding,
        #     self.genc_input,
        #     genc_hidden,
        #     gar_hidden,
        # )

    def forward(self, x):
        """Forward through the network"""

        # loss, accuracy, _, z = self.model(x)
        # return loss
        return self.model(x)
