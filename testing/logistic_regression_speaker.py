import sys
import os
import argparse
import torch
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from model import load_model, save_model
from data.loaders import librispeech_loader
from validation import validate_speakers
from modules.audio.speaker_loss import Speaker_Loss

#### pass configuration
from experiment import ex

## own modules
from data import loaders
from model import load_model


def train(args, context_model, loss, train_loader, optimizer, writer):
    total_step = len(train_loader)
    print_idx = 100

    total_i = 0
    for epoch in range(args.num_epochs):
        loss_epoch = 0
        acc_epoch = 0
        for i, (audio, filename, _, audio_idx) in enumerate(train_loader):

            starttime = time.time()

            loss.zero_grad()

            ### get latent representations for current audio
            model_input = audio.to(args.device)

            with torch.no_grad():
                z, c = context_model.module.model.get_latent_representations(model_input)

            c = c.detach()

            # forward pass
            total_loss, accuracies = loss.get_loss(
                model_input, z, c, filename, audio_idx
            )

            # Backward and optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            sample_loss = total_loss.item()
            accuracy = accuracies.item()

            writer.add_scalar("Loss/train_step", sample_loss, total_i)
            writer.add_scalar("Accuracy/train_step", accuracy, total_i)
            writer.flush()

            if i % print_idx == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Time (s): {:.1f}, Accuracy: {:.4f}, Loss: {:.4f}".format(
                        epoch + 1,
                        args.num_epochs,
                        i,
                        total_step,
                        time.time() - starttime,
                        accuracy,
                        sample_loss,
                    )
                )
                starttime = time.time()

            loss_epoch += sample_loss
            acc_epoch += accuracy
            total_i += 1

        writer.add_scalar("Loss/train_epoch", loss_epoch / total_step, epoch)
        writer.add_scalar("Accuracy/train_epoch", acc_epoch / total_step, epoch)
        writer.flush()

        # Sacred
        ex.log_scalar("train.loss", loss_epoch / total_step)
        ex.log_scalar("train.accuracy", acc_epoch / total_step)


def test(opt, context_model, loss, data_loader):
    loss.eval()

    accuracy = 0
    loss_epoch = 0

    with torch.no_grad():
        for i, (audio, filename, _, audio_idx) in enumerate(data_loader):

            loss.zero_grad()

            ### get latent representations for current audio
            model_input = audio.to(args.device)

            with torch.no_grad():
                z = context_model.module.forward_through_n_layers(model_input, 5)

            z = z.detach()

            # forward pass
            total_loss, step_accuracy = loss.get_loss(
                model_input, z, z, filename, audio_idx
            )

            accuracy += step_accuracy.item()
            loss_epoch += total_loss.item()

            if i % 10 == 0:
                print(
                    "Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}".format(
                        i, len(data_loader), loss_epoch / (i + 1), accuracy / (i + 1)
                    )
                )

    accuracy = accuracy / len(data_loader)
    loss_epoch = loss_epoch / len(data_loader)
    print("Final Testing Accuracy: ", accuracy)
    print("Final Testing Loss: ", loss_epoch)
    return loss_epoch, accuracy

@ex.automain
def main(_run, _log):
    args = argparse.Namespace(**_run.config)
    args.batch_size = 64
    args.num_epochs = 50
    args.learning_rate = 1e-3

    if len(_run.observers) > 1:
        out_dir = _run.observers[1].dir
    else:
        out_dir = _run.observers[0].dir

    args.out_dir = out_dir
    
    # set start time
    args.time = time.ctime()
    
    # Device configuration
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args.current_epoch = args.start_epoch

    # random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    ## load model
    context_model, optimizer = load_model(
        args, reload_model=True
    )
    context_model.eval()

    n_features = context_model.module.gar_hidden

    loss = Speaker_Loss(args, n_features, calc_accuracy=True)

    optimizer = torch.optim.Adam(loss.parameters(), lr=args.learning_rate)

    # load dataset
    train_loader, _, test_loader, _ = loaders.librispeech_loader(args)

    accuracy = 0
    
    # initialize TensorBoard
    tb_dir = os.path.join(out_dir, _run.experiment_info["name"])
    os.makedirs(tb_dir)
    writer = SummaryWriter(log_dir=tb_dir)

    try:
        # Train the model
        train(args, context_model, loss, train_loader, optimizer, writer)

        # Test the model
        result_loss, accuracy = test(args, context_model, loss, test_loader)

    except KeyboardInterrupt:
        print("Training interrupted, saving log files")
