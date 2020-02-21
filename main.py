import sys
import os
import argparse
import time
import torch
import numpy as np

# Apex for mixed-precision training
from apex import amp

# Sacred
from sacred import Experiment
from sacred.stflow import LogFileWriter
from sacred.observers import FileStorageObserver, MongoObserver

# TensorBoard
from torch.utils.tensorboard import SummaryWriter


from utils.yaml_config_hook import yaml_config_hook
from data.loaders import librispeech_loader

from model import load_model

#### pass configuration
ex = Experiment("contrastive-predictive-coding")

#### file output directory
ex.observers.append(FileStorageObserver("./logs"))

#### database output
ex.observers.append(
    MongoObserver().create(
        url=f"mongodb://admin:admin@localhost:27017/?authMechanism=SCRAM-SHA-1",
        db_name="db",
    )
)


@ex.config
def my_config():
    yaml_config_hook("./config/audio/config.yaml", ex)

    # override any settings here
    # start_epoch = 100
    # ex.add_config(
    #   {'start_epoch': start_epoch})


def train(args, model, optimizer, writer):

    # get datasets and dataloaders
    (train_loader, train_dataset, test_loader, test_dataset,) = librispeech_loader(
        args, num_workers=args.num_workers
    )

    total_step = len(train_loader)
    print_idx = 1

    start_time = time.time()
    for epoch in range(args.start_epoch, args.start_epoch + args.num_epochs):
        loss_epoch = 0
        for step, (audio, filename, _, start_idx) in enumerate(train_loader):

            if step % print_idx == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Time (s): {:.3f}".format(
                        epoch + 1,
                        args.start_epoch + args.num_epochs,
                        step,
                        total_step,
                        time.time() - start_time,
                    )
                )

            start_time = time.time()

            audio = audio.to(args.device)

            # forward
            loss = model(audio)

            # backward
            model.zero_grad()

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    loss.backward()
            else:
                loss.backward()

            optimizer.step()

            if step % print_idx == 0:
                print("\t \t Loss: \t \t {:.4f}".format(loss.item()))

            loss_epoch += loss

        avg_loss = loss_epoch / len(train_loader)
        writer.add_scalar("Train/loss", avg_loss)
        ex.add_scalar("train.loss", avg_loss)


@ex.automain
@LogFileWriter(ex)
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

    # initialize logger

    # load model
    model, optimizer = load_model(args)

    # set comment to experiment's name
    tb_dir = os.path.join(out_dir, _run.experiment_info["name"])
    os.makedirs(tb_dir)
    writer = SummaryWriter(log_dir=tb_dir)

    try:
        train(args, model, optimizer, writer)
    except KeyboardInterrupt:
        print("Interrupting training, saving logs")


if __name__ == "__main__":
    main()
