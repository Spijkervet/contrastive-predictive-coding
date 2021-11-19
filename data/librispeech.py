import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict
from glob import glob
import pandas as pd

csv_input = pd.read_csv(filepath_or_buffer='/groups/1/gcc50521/furukawa/musicnet_metadata.csv', sep=",")
genre_to_id = {
    'Solo Piano': 0, 'String Quartet': 1, 'Accompanied Violin': 2, 'Piano Quartet': 3, 'Accompanied Cello': 4,
    'String Sextet': 5, 'Piano Trio': 6, 'Piano Quintet': 7, 'Wind Quintet': 8, 'Horn Piano Trio': 9, 'Wind Octet': 10,
    'Clarinet-Cello-Piano Trio': 11, 'Pairs Clarinet-Horn-Bassoon': 12, 'Clarinet Quintet': 13, 'Solo Cello': 14,
    'Accompanied Clarinet': 15, 'Solo Violin': 16, 'Violin and Harpsichord': 17, 'Viola Quinte': 18, 'Solo Flute': 19
}
id_to_genre = {}
for idx, row in csv_input.iterrows():
    genre = row['ensemble']
    song_id = str(row['id'])
    id_to_genre[song_id] = genre


def default_loader(path):
    return torchaudio.load(path, normalization=False)


def default_flist_reader(root_dir):
    speaker_dict = defaultdict(list)
    item_list = []
    for index, x in enumerate(sorted(glob(os.path.join(root_dir, '*.npy')))):
        filename = x.split('/')[-1]
        speaker_id = id_to_genre[filename[:4]]
        item_list.append(speaker_id)
        speaker_dict[speaker_id].append(index)

    return speaker_dict, item_list


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

        self.file_list = sorted(glob(os.path.join(root, '*.npy')))
        self.speaker_dict, self.item_list = flist_reader(root)

        self.loader = loader
        self.audio_length = audio_length

    def __getitem__(self, index):
        filename = self.file_list[index]
        audio = torch.from_numpy(np.load(filename)).unsqueeze(0)
        speaker_id = self.item_list[index]

        # discard last part that is not a full 10ms
        max_length = audio.size(1) // 160 * 160

        start_idx = np.random.choice(
            np.arange(160, max_length - self.audio_length - 0, 160)
        )

        audio = audio[:, start_idx: start_idx + self.audio_length]

        # normalize the audio samples
        return audio, filename, speaker_id, start_idx

    def __len__(self):
        return len(self.file_list)

    def get_audio_by_speaker(self, speaker_id, batch_size):
        batch_size = min(len(self.speaker_dict[speaker_id]), batch_size)
        batch = torch.zeros(batch_size, 1, self.audio_length)
        for idx in range(batch_size):
            batch[idx, 0, :], _, _, _ = self.__getitem__(
                self.speaker_dict[speaker_id][idx]
            )

        return batch

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
