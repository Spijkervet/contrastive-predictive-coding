import sys
import argparse
import torch
import time
import os
import numpy as np

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

from model import load_model, save_model
from data.loaders import librispeech_loader
from validation import validate_speakers

#### pass configuration
from experiment import ex


from data import loaders, phone_dict


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.xavier_normal_(m.weight.data)


def train(
    args, train_dataset, phone_dict, context_model, model, optimizer, n_features, writer
):
    total_step = len(train_dataset.file_list)
    criterion = torch.nn.CrossEntropyLoss()
    total_i = 0
    for epoch in range(args.num_epochs):
        loss_epoch = 0
        acc_epoch = 0
        predicted_classes = torch.zeros(n_features).to(args.device)
        for i, k in enumerate(train_dataset.file_list):
            starttime = time.time()

            audio, filename = train_dataset.get_full_size_test_item(i)

            ### get latent representations for current audio
            model_input = audio.to(args.device)
            model_input = torch.unsqueeze(model_input, 0)


            targets = torch.LongTensor(phone_dict[filename])
            targets = targets.to(args.device).reshape(-1)

            with torch.no_grad():
                z, context = context_model.module.model.get_latent_representations(
                    model_input
                )

            context = context.detach()
            inputs = context.reshape(-1, n_features)

            # forward pass
            output = model(inputs)

            """ 
            The provided phone labels are slightly shorter than expected, 
            so we cut our predictions to the right length.
            Cutting from the front gave better results empirically.
            """
            output = output[-targets.size(0) :]  # output[ :targets.size(0)]

            loss = criterion(output, targets)

            # get distribution of predicted classes
            predicted = output.argmax(1)
            classes, counts = torch.unique(predicted, return_counts=True)
            predicted_classes[classes] += counts

            # calculate accuracy
            acc = (predicted == targets).sum().item() / targets.size(0)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sample_loss = loss.item()
            loss_epoch += sample_loss
            acc_epoch += acc

            writer.add_scalar("Loss/train_step", sample_loss, total_i)
            writer.add_scalar("Accuracy/train_step", acc, total_i)
            writer.flush()

            if i % 10 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Time (s): {:.1f}, Accuracy: {:.4f}, Loss: {:.4f}".format(
                        epoch + 1,
                        args.num_epochs,
                        i,
                        total_step,
                        time.time() - starttime,
                        acc,
                        sample_loss,
                    )
                )
            total_i += 1

        figure = plt.figure()
        plt.bar(range(predicted_classes.size(0)), predicted_classes.cpu().numpy())
        writer.add_figure("Class_distribution/global_step", figure, global_step=total_i)

        writer.add_scalar("Loss/train_epoch", loss_epoch / total_step, epoch)
        writer.add_scalar("Accuracy/train_epoch", acc_epoch / total_step, epoch)
        writer.flush()

        # Sacred
        ex.log_scalar("train.loss", loss_epoch / total_step)
        ex.log_scalar("train.accuracy", acc_epoch / total_step)

        # save current model state
        save_model(args, context_model, optimizer)
        args.current_epoch += 1


def test(args, test_dataset, phone_dict, context_model, model, optimizer, n_features):
    model.eval()

    total = 0
    correct = 0

    with torch.no_grad():
        for idx, k in enumerate(test_dataset.file_list):

            audio, filename = test_dataset.get_full_size_test_item(idx)

            model.zero_grad()

            ### get latent representations for current audio
            model_input = audio.to(args.device)
            model_input = torch.unsqueeze(model_input, 0)

            targets = torch.LongTensor(phone_dict[filename])

            with torch.no_grad():
                z, context = context_model.module.model.get_latent_representations(
                    model_input
                )

                context = context.detach()

                targets = targets.to(args.device).reshape(-1)
                inputs = context.reshape(-1, n_features)

                # forward pass
                output = model(inputs)

            output = output[-targets.size(0) :]

            # calculate accuracy
            _, predicted = torch.max(output.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            if idx % 1000 == 0:
                print(
                    "Step [{}/{}], Accuracy: {:.4f}".format(
                        idx, len(test_dataset.file_list), correct / total
                    )
                )

    accuracy = (correct / total) * 100
    print("Final Testing Accuracy: ", accuracy)
    return accuracy


@ex.automain
def main(_run, _log):
    args = argparse.Namespace(**_run.config)
    args.batch_size = 8
    args.num_epochs = 20

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

    # load self-supervised GIM model
    context_model, _ = load_model(args, reload_model=True)

    context_model.eval()

    # 41 different phones to differentiate
    n_classes = 41
    n_features = context_model.module.gar_hidden

    # create linear classifier
    model = torch.nn.Sequential(torch.nn.Linear(n_features, n_classes)).to(args.device)
    model.apply(weights_init)

    params = model.parameters()

    optimizer = torch.optim.Adam(params, lr=1e-4)

    # load dataset
    pd = phone_dict.load_phone_dict(args)
    _, train_dataset, _, test_dataset = loaders.librispeech_loader(args)

    accuracy = 0

    # initialize TensorBoard
    tb_dir = os.path.join(out_dir, _run.experiment_info["name"])
    os.makedirs(tb_dir)
    writer = SummaryWriter(log_dir=tb_dir)

    try:
        # Train the model
        train(
            args, train_dataset, pd, context_model, model, optimizer, n_features, writer
        )

        # Test the model
        accuracy = test(
            args, test_dataset, pd, context_model, model, optimizer, n_features
        )

    except KeyboardInterrupt:
        print("Interrupting training, saving model")
        save_model(args, context_model, optimizer)
