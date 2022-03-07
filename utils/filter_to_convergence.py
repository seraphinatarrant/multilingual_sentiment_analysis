import argparse
import os
from tqdm import tqdm

import pandas as pd

from model_utils import lang2convergence, compressed2convergence, multi_convergence, \
    compressed_multi_convergence


def setup_argparse():
    p = argparse.ArgumentParser()
    p.add_argument('--r', dest='results_dir')
    p.add_argument('--i', dest='input_files', nargs='+')
    p.add_argument('--compressed', action='store_true')
    p.add_argument('-l', dest='lang')
    return p.parse_args()


if __name__ == "__main__":
    args = setup_argparse()

    convergence_dict = compressed2convergence if args.compressed else lang2convergence
    multi_conv = compressed_multi_convergence if args.compressed else multi_convergence

    if args.results_dir:
        with os.scandir(args.results_dir) as source_dir:
            files = [f.path for f in source_dir if f.is_file() and f.name.endswith(".csv")]
    else:
        files = args.input_files

    for file in tqdm(files):
        print(f"Working on {file}")
        conv = multi_conv if "multi" in file else convergence_dict[args.lang]
        df = pd.read_csv(file)
        mask = df["steps"] == conv
        df = df[mask]
        new_filepath = os.path.splitext(file)[0] + "_convergence.csv"
        df.to_csv(new_filepath, index=False)

