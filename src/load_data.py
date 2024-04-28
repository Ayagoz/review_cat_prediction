import lmdb
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple
import torch
from torch.utils.data import Dataset

local_path: Path = Path(__file__).parent
logger: logging.Logger = logging.getLogger(__name__)


def get_target_labels(labels: List[str], index: List[str]) -> np.ndarray:
    print("-".join(str(labels[0]).split("-")[1:-1]))
    collect_labels = ["-".join(str(k).split("-")[1:-1])
                      for k in labels if "-".join(str(k).split("-")[1:-1]) in index]
    print(f"labels: {collect_labels}    \nlength of labels: {len(collect_labels)}")
    return collect_labels


def load_keys(path: str, filter_key: Optional[str] = None) -> List[bytes]:
    with lmdb.open(path, readonly=True) as env:
        with env.begin() as txn:
            my_list = [key for key, _ in txn.cursor() if key]

    if filter_key:
        my_list = [key for key in my_list if filter_key in str(key)]
    print(f"Length of embeds: {len(my_list)} \nembeds names: \n{my_list}")
    return my_list


def get_embed_by_key(path: str, key: str) -> Optional[np.ndarray]:
    output = None
    with lmdb.open(path, readonly=True) as env:
        value = env.begin().get(key)
        if value:
            output = np.frombuffer(bytearray(value), dtype=np.float32)
    return output


class LMDBDataset(Dataset):
    def __init__(self, path_data: str, path_target: str, filter_key: str = 'embed'):
        self.path_data = path_data
        self.keys = load_keys(path_data, filter_key=filter_key)
        self.target = pd.read_csv(path_target, index_col=0)
        self.labels = get_target_labels(self.keys, self.target.index)
        self.target = self.target.loc[self.labels]
        # self.values = np.stack([get_by_key(path_data, k) for k in self.keys])
        # self.y = get_targets(self.target, self.keys)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        value = get_embed_by_key(self.path_data, self.keys[index])
        target = self.target.loc[self.labels[index]]
        x = torch.from_numpy(value)
        y = torch.from_numpy(target.values.astype(np.float32))
        return x, y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Load Data')
    parser.add_argument('--path', type=str, required=True, help='Path to the LMDB database')
    parser.add_argument('--target', type=str, required=True, help='Path to the target CSV file')
    parser.add_argument('--filter_key', type=str, default='embed',
                        help='Filter key for loading keys')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset = LMDBDataset(args.path, args.target, filter_key=args.filter_key)
    for v, t in dataset:
        print(f"Shape of values: {v.shape}, shape of y_train: {t.shape}, target sum {t.sum()}")
