import argparse
import os
import sys

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plotnine import ggplot, aes, geom_count

from utils.model_utils import lang2convergence, compressed2convergence, multi_convergence, \
    compressed_multi_convergence, balanced_multi_convergence, compressed_balanced_multi_convergence

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
    else:
        sys.exit("Not a valid model type to check for convergence: {}".format(model_type))

    return steps


def setup_argparse():
    p = argparse.ArgumentParser()
    p.add_argument('-l', '--lang', dest='lang')
    p.add_argument('-o', dest='output_dir', default='analysis/plot_all/', help='output dir')
    p.add_argument('-pt', '--plot_type', choices=['swarm', 'scatter', 'bubble', 'violin'], default="swarm")

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
    file_insert = "_all_data"
    if args.plot_type != "scatter":
        for key, val in type2filepattern.items():
            _path, _filename = os.path.split(val)
            _file, _ext = os.path.splitext(_filename)
            new_filename = _file + file_insert + _ext
            new_val = os.path.join(_path, insert, new_filename)
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

    # break out by bias_type
    lang = args.lang if args.lang != "en_scrubbed" else "en"
    bias_types = lang2bias[lang]
    for bt in bias_types.keys():
        # get sub dataframe for one bias type
        mask = master_df["bias_type"] == bt
        this_df = master_df[mask]
        # set colour based on bias type
        if bt == "gender":
            colour = "mediumblue"
        elif bt == "race":
            colour = "firebrick"

        # set output filename
        outfile = os.path.join(args.output_dir, f"all_models_{args.lang}_{bt}.pdf")

        if args.plot_type == "bubble":
            myplot = ggplot(this_df, aes(x="model_type", y="performance_gap")) + geom_count(color=colour)
            myplot.save(outfile)
        else:
            if args.plot_type == "violin":
                myplot = sns.violinplot(data=this_df, x="model_type", y="performance_gap", color=colour, order=type_order)
            else:
                myplot = sns.stripplot(data=this_df, x="model_type", y="performance_gap", color=colour, order=type_order, jitter=True, dodge=True)
                if args.plot_type == "scatter":
                    # stat sig
                    significance_mask = this_df["statistical_significance"] > (0.05/num_models)
                    sig_df = this_df[significance_mask]
                    myplot = sns.stripplot(data=sig_df, x="model_type", y="performance_gap", color="black", s=100, order=type_order, marker="x")
            myplot.set(ylabel=None)#, yticklabels=[])
            myplot.set(xlabel=None)
            #myplot.tick_params(left=False)
            myplot.axhline(0.0, linestyle=":", color="gray")
            plt.xticks(rotation=90)
            plt.ylim(*y_axis)
            plt.subplots_adjust(bottom=0.28)
            plt.savefig(outfile)
            plt.clf()
            
