import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict
from glob import glob


def default_loader(path):
    return torchaudio.load(path, normalization=False)


def default_flist_reader(flist):
    item_list = []
    speaker_dict = defaultdict(list)
    index = 0
    with open(flist, "r") as rf:
        for line in rf.readlines():
            speaker_id, dir_id, sample_id = line.replace("\n", "").split("-")
            item_list.append((speaker_id, dir_id, sample_id))
            speaker_dict[speaker_id].append(index)
            index += 1

    return item_list, speaker_dict


class LibriDataset(Dataset):
    def __init__(
        self,
        opt,
        root,
        flist,
        audio_length=20480,
        flist_reader=default_flist_reader,
        loader=default_loader,
    ):
        self.root = root
        self.opt = opt

        self.file_list = glob(os.path.join(root, '*.npy'))

        self.loader = loader
        self.audio_length = audio_length

    def __getitem__(self, index):
        filename = self.file_list[index]
        audio = torch.from_numpy(np.load(filename)).unsqueeze(0)

        # discard last part that is not a full 10ms
        max_length = audio.size(1) // 160 * 160

        start_idx = np.random.choice(
            np.arange(160, max_length - self.audio_length - 0, 160)
        )

        audio = audio[:, start_idx : start_idx + self.audio_length]

        # normalize the audio samples
        return audio, filename, start_idx

    def __len__(self):
        return len(self.file_list)

    def get_full_size_test_item(self, index):
        """
        get audio samples that cover the full length of the input files
        used for testing the phone classification performance
        """
        filename = self.file_list[index]
        audio = torch.from_numpy(np.load(filename)).unsqueeze(0)

        ## discard last part that is not a full 10ms
        max_length = audio.size(1) // 160 * 160
        audio = audio[:max_length]

        return audio, filename
