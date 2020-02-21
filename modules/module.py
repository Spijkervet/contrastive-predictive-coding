import torch
from .encoder import Encoder
from .autoregressor import Autoregressor
from .infonce import InfoNCE


class Module(torch.nn.Module):
    def __init__(
        self, args, strides, filter_sizes, padding, genc_input, genc_hidden, gar_hidden,
    ):
        super(Module, self).__init__()

        """
        First, a non-linear encoder genc maps the input sequence of observations xt to a
        sequence of latent representations zt = genc(xt), potentially with a lower temporal resolution.
        """
        self.encoder = Encoder(genc_input, genc_hidden, strides, filter_sizes, padding,)

        """
        We then use a GRU RNN [17] for the autoregressive part of the model, gar with 256 dimensional hidden state.
        """
        self.autoregressor = Autoregressor(args, input_dim=genc_hidden, hidden_dim=gar_hidden)

        self.loss = InfoNCE(args, gar_hidden, genc_hidden)

    def forward(self, x):
        """
        Calculate latent representation of the input with the encoder and autoregressor
        :param x: inputs (B x C x L)
        :return: loss - calculated loss
                accuracy - calculated accuracy
                z - latent representation from the encoder (B x L x C)
                c - latent representation of the autoregressor  (B x C x L)
        """

        # calculate latent represention from the encoder
        z = self.encoder(x)
        z = z.permute(0, 2, 1)  # swap L and C

        # calculate latent representation from the autoregressor
        c = self.autoregressor(z)

        loss, accuracy = self.loss.get(x, z, c)
        return loss, accuracy, z, c

