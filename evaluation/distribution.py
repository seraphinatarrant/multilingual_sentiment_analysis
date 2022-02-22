import argparse
import os

import numpy as np
import yaml

import pandas as pd
from scipy.stats import wasserstein_distance
import sys
import matplotlib.pyplot as plt
import seaborn as sns

def setup_argparse():
    p = argparse.ArgumentParser()
    p.add_argument('-l', dest='lang', default='ja',
                   help='a the lang to do the analysis on')
    p.add_argument('-o', dest='output', default='analysis/label_distributions/')
    p.add_argument('-r', dest='results_dir', default="results/{}/full_output/")
    return p.parse_args()

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
    for key in patterns.keys():
        df = pd.read_csv(os.path.join(prefix, patterns[key]))
        vals = np.concatenate([df["label_1"].to_numpy(), df["label_2"].to_numpy()])
        print(len(vals))
        myplot = sns.histplot(data=vals, discrete=True)
        output_name = f"{args.lang}_predictions.pdf" if key =="mono" else f"multi_{args.lang}_predictions.pdf"
        plt.savefig(os.path.join(args.output, output_name))
        plt.clf()
        dists.append(vals)

    print(wasserstein_distance(dists[0], dists[1]))


