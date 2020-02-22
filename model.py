import torch
from modules import AudioModel


def audio_model(args):
    strides = [5, 4, 2, 2, 2]
    filter_sizes = [10, 8, 4, 4, 4]
    padding = [2, 2, 2, 2, 1]
    genc_hidden = 512
    gar_hidden = 256

    model = AudioModel(
        args,
        strides=strides,
        filter_sizes=filter_sizes,
        padding=padding,
        genc_hidden=genc_hidden,
        gar_hidden=gar_hidden,
    )
    return model


def load_model(args):

    if args.experiment == "audio":
        model = audio_model(args)
    else:
        raise NotImplementedError

    model = model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Install the apex package from https://www.github.com/nvidia/apex to use fp16 for training"
            )

        print("### USING FP16 ###")
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level
        )

    print("Using {} GPUs".format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    return model, optimizer
