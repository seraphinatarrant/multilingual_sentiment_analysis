import argparse
import os
import sys

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils.model_utils import lang2convergence, compressed2convergence, multi_convergence, \
    compressed_multi_convergence, balanced_multi_convergence, compressed_balanced_multi_convergence

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
    elif model_type == "mono_compressed":
        steps = compressed2convergence[lang]
    elif model_type == "multi_xl_compressed":
        steps = compressed_multi_convergence
    elif model_type == "multi_xl_balanced":
        steps = balanced_multi_convergence
    elif model_type == "multi_xl_compressed_balanced":
        steps = compressed_balanced_multi_convergence
    else:
        sys.exit("Not a valid model type to check for convergence: {}".format(model_type))

    return steps


def setup_argparse():
    p = argparse.ArgumentParser()
    p.add_argument('-l', '--lang', dest='lang')
    p.add_argument('-o', dest='output_dir', default='analysis/plot_all/', help='output dir')
    p.add_argument('-pt', '--plot_type', choices=['swarm', 'scatter'], default="swarm")

    return p.parse_args()


if __name__ == "__main__":
    args = setup_argparse()

    print(args)

    type_order = ["baseline", "mono", "mono_c", "multi_xl",
                  "multi_xl_c", "multi_xl_b", "multi_xl_c_b"]

    type2filepattern = {
        "baseline": "results/baseline/{}/{}_ensemble.csv",
        "mono": "results/{}/{}_ensemble.csv",
        "multi_xl": "results/{}/multi+en_{}_ensemble.csv",
        "mono_c": "results/compressed/{}/{}_ensemble.csv",
        "multi_xl_c": "results/compressed/{}/multi+en_{}_ensemble.csv",
        "multi_xl_b": "results/balanced/{}/multi+en_{}_ensemble.csv",
        "multi_xl_c_b": "results/balanced/compressed/{}/multi+en_{}_ensemble.csv",
    }

    insert = "full_output"
    if args.plot_type == "strip":
        for key, val in type2filepattern.items():
            _path, _filename = os.path.split(val)
            new_val = os.path.join(_path, insert, _filename)
            type2filepattern[key] = new_val

        y_axis = (-4, 4)

    else:
        y_axis = (-0.4, 0.8) # set empirically based on average gaps

    num_models = len(type2filepattern)


    master_df = pd.DataFrame()

    for model_type, file_pattern in type2filepattern.items():
            print(model_type)
            try:
                infile = file_pattern.format(args.lang, args.lang)
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
            master_df = master_df.append(df)
            #myplot = sns.scatterplot(data=df, x="steps", y="performance_gap")

    mask = master_df["bias_type"] == "rank"
    master_df = master_df[~mask]

    myplot = sns.swarmplot(data=master_df, x="model_type", y="performance_gap", hue="bias_type", order=type_order)

    if args.plot_type == "scatter":
        # stat sig
        significance_mask = master_df["statistical_significance"] > (0.05/num_models)
        sig_df = master_df[significance_mask]
        myplot = sns.swarmplot(data=sig_df, x="model_type", y="performance_gap", color="black", s=100, order=type_order, marker="x")
    #myplot.set(ylabel=None, yticklabels=[])
    #myplot.tick_params(left=False)
    myplot.axhline(0.0, linestyle=":", color="gray")
    plt.xticks(rotation=90)
    plt.ylim(*y_axis)
    plt.savefig(os.path.join(args.output_dir, f"all_models_{args.lang}.pdf"))
    #plt.clf()
    


            
