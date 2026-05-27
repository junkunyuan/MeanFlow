"""Convert HuggingFace ImageNet-1k parquet shards to an LMDB compatible with
main_cache.py::LMDBImageNetReader.

LMDB layout (matches image2lmdb.py):
    key f'{idx}'        -> pickle({'image': <jpeg bytes>, 'label': <int>})
    key 'num_samples'   -> str
    key 'num_classes'   -> str
    key 'class_names'   -> pickle(list[str])  (WordNet ids from classes.py)
"""
import argparse
import importlib.util
import os
import pickle
import sys
from glob import glob

import lmdb
import pyarrow.parquet as pq
from tqdm import tqdm


def load_class_names(classes_py_path):
    spec = importlib.util.spec_from_file_location("imagenet_classes", classes_py_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.IMAGENET2012_CLASSES.keys())


def parquet_to_lmdb(parquet_dir, split, lmdb_path, map_size_gb, class_names):
    pattern = os.path.join(parquet_dir, f"{split}-*.parquet")
    shards = sorted(glob(pattern))
    if not shards:
        raise FileNotFoundError(f"No parquet files match {pattern}")

    print(f"split={split}: found {len(shards)} parquet shards")
    print(f"writing to {lmdb_path} (map_size={map_size_gb} GB)")

    os.makedirs(os.path.dirname(lmdb_path) or ".", exist_ok=True)
    map_size = map_size_gb * 1024 ** 3
    env = lmdb.open(lmdb_path, map_size=map_size)

    idx = 0
    try:
        for shard_path in tqdm(shards, desc="shards"):
            table = pq.read_table(shard_path, columns=["image", "label"])
            images = table.column("image").to_pylist()
            labels = table.column("label").to_pylist()

            with env.begin(write=True) as txn:
                for img, label in zip(images, labels):
                    img_bytes = img["bytes"]
                    if img_bytes is None:
                        # parquet row stored only the path — skip; HF ImageNet-1k
                        # ships bytes inline so this shouldn't normally trigger.
                        print(f"warning: row {idx} in {shard_path} has no bytes, skipping")
                        continue
                    entry = {"image": img_bytes, "label": int(label)}
                    txn.put(f"{idx}".encode(), pickle.dumps(entry))
                    idx += 1

        with env.begin(write=True) as txn:
            txn.put(b"num_samples", str(idx).encode())
            txn.put(b"num_classes", str(len(class_names)).encode())
            txn.put(b"class_names", pickle.dumps(class_names))
    finally:
        env.sync()
        env.close()

    print(f"done: wrote {idx} samples to {lmdb_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet_dir", default="/opt/tiger/toys/ILSVRC/imagenet-1k/data")
    p.add_argument("--classes_py", default="/opt/tiger/toys/ILSVRC/imagenet-1k/classes.py")
    p.add_argument("--split", required=True, choices=["train", "validation", "test"])
    p.add_argument("--lmdb_path", required=True)
    p.add_argument("--map_size_gb", type=int, default=300,
                   help="LMDB max size in GB (train ~140, val ~6)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    class_names = load_class_names(args.classes_py)
    parquet_to_lmdb(
        parquet_dir=args.parquet_dir,
        split=args.split,
        lmdb_path=args.lmdb_path,
        map_size_gb=args.map_size_gb,
        class_names=class_names,
    )
