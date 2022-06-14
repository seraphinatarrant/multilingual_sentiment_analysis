import argparse
import os
import sys
from tqdm import tqdm
import numpy as np

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from plotnine import ggplot, aes, geom_count
from sklearn import metrics

from utils.model_utils import lang2convergence, compressed2convergence, multi_convergence, \
    compressed_multi_convergence, balanced_multi_convergence, compressed_balanced_multi_convergence, mono_multi2convergence
from utils.graph_utils import get_cmap_midpoint, shifted_colormap


from evaluation.create_eval_set import lang2bias

def get_convergence_by_type(model_type, lang):
    """
    multilingual models are zero shot to training data so the convergence numbers are the same
    regardless of language applied to. Monolingual models are different per language so they have
    a dict to query convergence steps by language
    """
    if model_type == "mono":
        steps = lang2convergence[lang]
    elif model_type == "multi_xl":
        steps = multi_convergence
    elif model_type == "mono_c":
        steps = compressed2convergence[lang]
    elif model_type == "multi_xl_c":
        steps = compressed_multi_convergence
    elif model_type == "multi_xl_b":
        steps = balanced_multi_convergence
    elif model_type == "multi_xl_c_b":
        steps = compressed_balanced_multi_convergence
    elif model_type == "multi_on_mono":
        steps = mono_multi2convergence[lang]
    else:
        sys.exit("Not a valid model type to check for convergence: {}".format(model_type))

    return steps


def setup_argparse():
    p = argparse.ArgumentParser()
    p.add_argument('-l', '--langs', dest='langs', nargs='+', default=["ja", "en", "zh", "es", "zh"])
    p.add_argument('-o', dest='output_dir', default='results/dfs/', help='output dir')
    p.add_argument('--polarity', action='store_true', help='use results that have been preconverted to polarity')

    return p.parse_args()


if __name__ == "__main__":
    args = setup_argparse()

    print(args)

    type2filepattern = {
        "baseline": "results/baseline/{}/{}_ensemble.csv",
        "mono": "results/{}/{}_ensemble.csv",
        "multi_on_mono": "results/mono_multi/{}/multi+{}_{}_ensemble.csv",
        "multi_xl": "results/{}/multi+en_{}_ensemble.csv",
        "mono_c": "results/compressed/{}/{}_ensemble.csv",
        "multi_xl_c": "results/compressed/{}/multi+en_{}_ensemble.csv",
        "multi_xl_b": "results/balanced/{}/multi+en_{}_ensemble.csv",
        "multi_xl_c_b": "results/balanced/compressed/{}/multi+en_{}_ensemble.csv",
    }

    insert = "full_output"
    file_insert = "_all_data_emotion_polarity" if args.polarity else "_all_data"
    if args.plot_type != "scatter":
        for key, val in type2filepattern.items():
            _path, _filename = os.path.split(val)
            _file, _ext = os.path.splitext(_filename)
            new_filename = _file + file_insert + _ext
            new_val = os.path.join(_path, insert, new_filename)
            type2filepattern[key] = new_val

    num_models = len(type2filepattern)

    # This all just makes the correct dataframes
    master_df = pd.DataFrame()

    print("Gathering dataframes...")
    for model_type, file_pattern in type2filepattern.items():
        print(model_type)
        for lang in args.langs:
            try:
                infile = file_pattern.format(args.lang, args.lang) if model_type != "multi_on_mono" else file_pattern.format(args.lang, args.lang, args.lang)
                df = pd.read_csv(infile)
            except:
                print("Couldn't read in file, may not exist: {}".format(infile), file=sys.stderr)
                continue
            if args.lang == "en_scrubbed":
                df["lang"] = "en_s"
            if model_type != "baseline":
                convergence_steps = get_convergence_by_type(model_type, args.lang)
                mask = df["steps"] == convergence_steps
                df = df[mask]
                df["model_type"] = model_type
                master_df = master_df.append(df, ignore_index=True)

        mask = master_df["bias_type"] == "rank" # artifact of a tested bias type in Japanese that no longer use
        master_df = master_df[~mask]
        master_df.to_pickle(os.path.join(args.output_dir, "master_df.pkl"))

