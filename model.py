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

    cpc = cpc.to(args.device)


    optimizer = optim.Adam(cpc.parameters(), lr=args.learning_rate)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Install the apex package from https://www.github.com/nvidia/apex to use fp16 for training"
            )

        print("### USING FP16 ###")
        cpc, optimizer = amp.initialize(
            cpc, optimizer, opt_level=args.fp16_opt_level
        )

    return cpc, optimizer
