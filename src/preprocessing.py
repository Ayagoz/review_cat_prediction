import lmdb
import json
import logging
from typing import List, Tuple, Union, Any
from pathlib import Path

import argparse
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

local_path = Path(__file__).parent
logger = logging.getLogger(__name__)


def create_embeeding(texts: List[str], model, mean: bool = False, pick_random: bool = False, size: int = 1) -> Tuple[Union[np.ndarray, None], List[int]]:
    final_embed = None
    idx = list(range(len(texts)))
    if pick_random:
        idx = np.random.choice(len(texts), size=size, replace=False)
        final_embed = np.mean([model.encode(texts[one]) for one in idx], axis=0)
    if mean:
        embeddings = []
        for one in texts:
            embed = model.encode(one)
            embeddings.append(embed)
        final_embed = np.mean(embeddings, axis=0)
    if final_embed is None:
        final_embed = model.encode(texts)

    return final_embed, idx


def create_db(path: str, idx: List[Any], model, dataset: str, mean: bool, pick_random: bool, size: int = 1) -> None:
    with lmdb.open(path, map_size=int(1e12)) as env:
        with env.begin(write=True) as txn:
            for i, t in enumerate(idx):
                all_texts = []
                with open(dataset, "r") as f:
                    for line in f.readlines():
                        record = json.loads(line)
                        if record['business_id'] == t:
                            all_texts.append(record['text'])
                final_embed, idx = create_embeeding(
                    all_texts, model, mean=mean, pick_random=pick_random, size=size)
                all_texts = [all_texts[i] for i in idx]
                embed = f"embed-{t}-{i}"
                texts = f"texts-{t}-{i}"
                txn.put(embed.encode(), np.array(final_embed).tobytes())
                if len(all_texts) >= 20:
                    for k in range(0, len(all_texts) // 20 + 1):
                        texts = f"texts-{t}-{i}-{k}"
                        txn.put(texts.encode(), np.array(
                            "\n".join(all_texts[k * 20: (k + 1) * 20])).tobytes())
                else:
                    txn.put(texts.encode(), np.array("\n".join(all_texts)).tobytes())

                print(f"Sample {embed}, num of review {len(all_texts)}")
            txn.put('nsamples'.encode(), np.array([i + 1]).tobytes())
    print(f"Total number of sample: {i + 1}")


def prepare_data(path_target: str, path_data: str, model, save_path: str, N: int, mean: bool, pick_random: bool, size: int = 1, shuffle: bool = False) -> None:
    target = pd.read_csv(path_target, index_col=0)
    idx = list(target.index)
    del target

    if shuffle:
        np.random.shuffle(idx)
    create_db(save_path, idx[:N], model=model, dataset=path_data, mean=mean,
              pick_random=pick_random, size=size)


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocessing script')
    # required arguments: data and target files + save db path
    parser.add_argument('--target', type=str, help='Path to the target file', required=True)
    parser.add_argument('--data', type=str, help='Path to the data file', required=True)
    parser.add_argument('--save_path', type=str, help='Path to save the database', required=True)

    # model type and number of samples
    parser.add_argument('--model', type=str, help='Path to the model file',
                        required=False, default='sentence-transformers/all-mpnet-base-v2')
    parser.add_argument('--N', type=int, help='Number of samples', required=False, default=1)

    # mean or random aggregation type for embeddings
    parser.add_argument('--mean', action='store_true',
                        help='Use mean embedding', required=False, default=True)
    parser.add_argument('--pick_random', action='store_true',
                        help='Pick random samples', required=False, default=False)
    parser.add_argument('--size', type=int, help='Size of random samples',
                        required=False, default=1)
    parser.add_argument('--shuffle', action='store_true',
                        help='Shuffle the data', required=False, default=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    Path(args.save_path).mkdir(exist_ok=True)
    model = SentenceTransformer(args.model)
    prepare_data(path_target=args.target, path_data=args.data, model=model,
                 save_path=args.save_path, N=args.N, mean=args.mean,
                 pick_random=args.pick_random, size=args.size, shuffle=args.shuffle)
