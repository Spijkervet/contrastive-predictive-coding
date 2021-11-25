import os
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# from tsnecuda import TSNE


def plot_tsne(args, embedding, labels, epoch, step):
    fp = os.path.join(args.out_dir, "tsne", "{}-{}.png".format(epoch, step))
    if not os.path.exists(os.path.dirname(fp)):
        os.makedirs(os.path.dirname(fp))

    figure = plt.figure(figsize=(8, 8), dpi=120)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels.ravel())
    plt.axis("off")
    plt.savefig(fp, bbox_inches="tight")
    return figure


def tsne(args, features):
    embedding = TSNE().fit_transform(features)
    return embedding


def validate_speakers(args, dataset, model, optimizer, epoch, step, global_step, writer):

    max_speakers = 20
    batch_size = 40
    input_size = (args.batch_size, 1, 20480)

    model.eval()
    with torch.no_grad():
        # DataParallel wraps model in module
        model = model.module.model
        latent_rep_size, latent_rep_len = model.get_latent_size(input_size)
        features = torch.zeros(
            max_speakers, batch_size, latent_rep_size * latent_rep_len
        ).to(args.device)
        labels = torch.zeros(max_speakers, batch_size).to(args.device)

        for idx, speaker_idx in enumerate(dataset.speaker_dict):
            if idx == 20:
                break

            model_in = dataset.get_audio_by_speaker(speaker_idx, batch_size=batch_size)
            model_in = model_in.to(args.device)
            z, c = model.get_latent_representations(model_in)

            z_repr = z.permute(0, 2, 1)
            c_repr = c.permute(0, 2, 1)

            features[idx, :, :] = c_repr.reshape((batch_size, -1))
            labels[idx, :] = idx

    features = features.reshape(features.size(0) * features.size(1), -1).cpu()
    labels = labels.reshape(-1, 1).cpu().numpy()

    embedding = tsne(args, features)
    figure = plot_tsne(args, embedding, labels, epoch, step)

    # add to TensorBoard
    writer.add_embedding(features, metadata=labels, global_step=global_step)
    writer.add_figure("TSNE", figure, global_step=global_step)
    writer.flush()

    out_dir = os.path.join(args.out_dir, "tsne")
    torch.save(features, os.path.join(out_dir, "features-{}-{}.pt".format(epoch, step)))
    torch.save(labels, os.path.join(out_dir, "labels-{}-{}.pt".format(epoch, step)))

    model.train()
