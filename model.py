import torch.optim as optim
from modules import CPC


def load_model(args):
    strides = [5, 4, 2, 2, 2]
    filter_sizes = [10, 8, 4, 4, 4]
    padding = [2, 2, 2, 2, 1]
    genc_hidden = 512
    gar_hidden = 256

    cpc = CPC(
        args,
        strides=strides,
        filter_sizes=filter_sizes,
        padding=padding,
        genc_hidden=genc_hidden,
        gar_hidden=gar_hidden,
    )

    optimizer = optim.Adam(cpc.parameters(), lr=args.learning_rate)

    return cpc, optimizer
