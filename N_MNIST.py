import torch

from os import listdir
from os.path import isfile, join

import numpy as np

class N_MNIST(torch.utils.data.Dataset):
    def __init__(self, split):

        root = "./data"
        sub_root = f"{root}/{split}"

        self.data = []

        for i in range(10):
            path = f"{sub_root}/{i}"

            for file in listdir(path):
                file_path = join(path, file)
                if isfile(file_path): 
                    self.data.append((file_path, i))

        self.T = 100

        
    def __len__(self):
        return len(self.data)
    

    def read_nmnist_file(self, file_path):
        data = np.fromfile(file_path, dtype=np.uint8)
        data = data.reshape(-1, 5)

        x = data[:, 0]
        y = data[:, 1]

        polarity = (data[:, 2] >> 7) & 1

        timestamp = (
            ((data[:, 2] & 0x7F) << 16) |
            (data[:, 3] << 8) |
            data[:, 4]
        )

        events = np.stack([x, y, timestamp, polarity], axis=1)

        return torch.tensor(events)

    def events_to_frames(self, events, H=34, W=34):
        x = events[:, 0].long()
        y = events[:, 1].long()
        t = events[:, 2]
        p = events[:, 3].long()

        t_min = t.min()
        t_max = t.max()

        # Normalize timestamps → [0, T-1]
        t_norm = ((t - t_min).float() / (t_max - t_min + 1e-6) * (self.T - 1)).long()

        frames = torch.zeros(self.T, H * 2, W * 2)

        # Valid coordinates
        mask = (x < W) & (y < H)
        x = x[mask]
        y = y[mask]
        p = p[mask]
        t_norm = t_norm[mask]

        # Positive polarity
        pos_mask = p == 1
        frames[t_norm[pos_mask], y[pos_mask], x[pos_mask]] = 1.0

        # Negative polarity → shifted
        neg_mask = p == 0
        frames[
            t_norm[neg_mask],
            y[neg_mask] + H,
            x[neg_mask] + W
        ] = 1.0

        return frames

    def __getitem__(self, index):
        file_path, target = self.data[index]

        events = self.read_nmnist_file(file_path)
        frames = self.events_to_frames(events)

        frames = frames.reshape((self.T, -1))

        return frames, target