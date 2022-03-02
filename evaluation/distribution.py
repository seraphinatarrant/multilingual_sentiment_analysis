import argparse
import os

import numpy as np

import pandas as pd
from scipy.stats import wasserstein_distance
import sys
import matplotlib.pyplot as plt
import seaborn as sns

from utils.model_utils import lang2convergence, compressed2convergence, multi_convergence, compressed_multi_convergence

def setup_argparse():
    p = argparse.ArgumentParser()
    p.add_argument('-l', dest='lang', default='ja',
                   help='a the lang to do the analysis on')
    p.add_argument('-o', dest='output', default='analysis/label_distributions/')
    p.add_argument('-r', dest='results_dir', default="results/{}/full_output/")
    p.add_argument('--convergence', action='store_true', help="only graph models at convergence")
    p.add_argument('--compressed', action='store_true', help='only used if convergence is true')
    p.add_argument('--baseline', action='store_true')
    return p.parse_args()


def plot_and_save(vals, out_name):
    myplot = sns.histplot(data=vals, discrete=True)
    plt.savefig(out_name)
    plt.clf()


if __name__ == "__main__":
    args = setup_argparse()

    # get distribution from results file
    # calculate wasserstein distance
    # label_1,label_2
    prefix = args.results_dir.format(args.lang)
    patterns = {
        "mono": "{}_ensemble_all_data.csv".format(args.lang),
        "multi": "multi+en_{}_ensemble_all_data.csv".format(args.lang)
    }

    dists = []

    if args.baseline:
        # Will be much simpler since all baseline results are already at convergence, this is mostly duplicate code
        prefix = args.results_dir.format(f'baseline/{args.lang}')
        pattern = patterns["mono"]

        df = pd.read_csv(os.path.join(prefix, pattern))
        vals = np.concatenate([df["label_1"].to_numpy(), df["label_2"].to_numpy()])
        print(len(vals))
        output_name = f"{args.lang}_predictions.pdf"
        plot_and_save(vals, os.path.join(args.output, output_name))


    if args.convergence:
        if args.compressed:
            convergence_dict = compressed2convergence
            multi_convergence = compressed_multi_convergence
        else:
            convergence_dict = lang2convergence

    for key in patterns.keys():
        df = pd.read_csv(os.path.join(prefix, patterns[key]))
        if args.convergence:
            if key == "multi":
                mask = df["steps"] == multi_convergence
            else:
                mask = df["steps"] == convergence_dict[args.lang]
            df = df[mask]
        vals = np.concatenate([df["label_1"].to_numpy(), df["label_2"].to_numpy()])
        print(len(vals))
        output_name = f"{args.lang}_predictions.pdf" if key == "mono" else f"multi_{args.lang}_predictions.pdf"
        plot_and_save(vals, os.path.join(args.output, output_name))
        dists.append(vals)

    print(wasserstein_distance(dists[0], dists[1]))


