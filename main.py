import os
import argparse
import time
import torch
import numpy as np
from datetime import datetime
# Apex for mixed-precision training
from apex import amp


# TensorBoard
from torch.utils.tensorboard import SummaryWriter


from model import load_model
from data.loaders import librispeech_loader
from validation import validate_speakers

#### pass configuration
from experiment import ex

def train(args, model, optimizer, writer):

    # get datasets and dataloaders
    (train_loader, train_dataset, test_loader, test_dataset,) = librispeech_loader(
        args, num_workers=args.num_workers
    )

    total_step = len(train_loader)
    print_idx = 100

    # at which step to validate training
    validation_idx = 1000

    start_time = time.time()
    global_step = 0
    for epoch in range(args.start_epoch, args.start_epoch + args.num_epochs):
        loss_epoch = 0
        for step, (audio, filename, _, start_idx) in enumerate(train_loader):

            start_time = time.time()

            if step % validation_idx == 0:
                validate_speakers(args, train_dataset, model, optimizer, epoch, step, writer)

            audio = audio.to(args.device)

            # forward
            loss = model(audio)

            # backward, depending on mixed-precision
            model.zero_grad()
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            optimizer.step()

            if step % print_idx == 0:
                examples_per_second = args.batch_size / (time.time() - start_time)
                print(
                    "[Epoch {}/{}] Train step {:04d}/{:04d} \t Examples/s = {:.2f} \t "
                    "Loss = {:.4f} \t Time/step = {:.4f}".format(
                        epoch,
                        args.num_epochs,
                        step,
                        len(train_loader),
                        examples_per_second,
                        loss,
                        time.time() - start_time
                    )
                )

            writer.add_scalar("Loss/train_step", loss, global_step)
            loss_epoch += loss
            global_step += 1

        avg_loss = loss_epoch / len(train_loader)
        writer.add_scalar("Loss/train", avg_loss, epoch)
        ex.log_scalar("loss.train", avg_loss, epoch)


@ex.automain
def main(_run, _log):
    args = argparse.Namespace(**_run.config)

    if len(_run.observers) > 1:
        out_dir = _run.observers[1].dir
    else:
        out_dir = _run.observers[0].dir

    args.out_dir = out_dir

    # set start time
    args.time = time.ctime()

    # Device configuration
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load model
    model, optimizer = load_model(args)

    # set comment to experiment's name
    tb_dir = os.path.join(out_dir, _run.experiment_info["name"])
    os.makedirs(tb_dir)
    writer = SummaryWriter(log_dir=tb_dir)

    try:
        train(args, model, optimizer, writer)
    except KeyboardInterrupt:
        print("Interrupting training, saving model")


if __name__ == "__main__":
    main()
