import argparse
import os
import sys
import ipdb
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
    p.add_argument('-l', '--lang', dest='lang')
    p.add_argument('-o', dest='output_dir', default='analysis/plot_all/', help='output dir')
    p.add_argument('-pt', '--plot_type', choices=['heatmap', 'scatter', 'bubble', 'violin', "errbars"],
                   default="errbars")
    p.add_argument('--only_models', nargs='+', default=None, help='restrict to only this list of models') 
    p.add_argument('--polarity', action='store_true', help='use results that have been preconverted to polarity')
    p.add_argument('--include_gold', action='store_true', help='if doing polarity, can also include '
                                                               'gold results, currently only works with heatmap')

    return p.parse_args()


if __name__ == "__main__":
    args = setup_argparse()

    print(args)

    type_order = ["baseline", "mono", "multi_on_mono", "mono_c", "multi_xl",
                  "multi_xl_c", "multi_xl_b", "multi_xl_c_b"]

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

    if args.plot_type == "scatter" or args.plot_type == "errbars":
        if args.polarity:
            y_axis = (-0.2, 0.3)
            xspan = 0.05
        else:
            y_axis = (-0.4, 0.8) # set empirically based on average gaps
            xspan = 0.11
    else:
        y_axis = (-4, 4)

    num_models = len(type2filepattern)

    # This all just makes the correct dataframes
    master_df = pd.DataFrame()
    if args.only_models:
        type_order = args.only_models
    print("Gathering dataframes...")
    for model_type, file_pattern in type2filepattern.items():
        if args.only_models:
            if model_type not in args.only_models:
                continue
            
        print(model_type)
        try:
            infile = file_pattern.format(args.lang, args.lang) if model_type != "multi_on_mono" else file_pattern.format(args.lang, args.lang, args.lang)
            df = pd.read_csv(infile)
        except:
            print("Couldn't read in file, may not exist: {}".format(infile), file=sys.stderr)
            type_order.remove(model_type) # make sure don't use later
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

    ## This is where the plotting happens
    # break out by bias_type
    lang = args.lang if args.lang != "en_scrubbed" else "en"
    bias_types = lang2bias[lang]
    print("Plotting...")
    for bt in bias_types.keys():
        # get sub dataframe for one bias type
        mask = master_df["bias_type"] == bt
        this_df = master_df[mask]
        # set colour based on bias type
        if bt == "gender":
            colour = "mediumblue"
        elif bt == "race":
            colour = "firebrick"

        # if doing a confusion matrix heatmap, then need to also do a separate graph for each model
        if args.plot_type == "heatmap":
            for model_type in type_order:
                print(model_type)
                labels = range(3) if args.polarity else range(1, 6)
                mask = this_df["model_type"] == model_type
                model_df = this_df[mask]
                if len(model_df.index) == 0:
                    print(f"Skipping {model_type}, no data for it")
                    continue
                # to use for labelling later
                bias_cat_1 = list(set(model_df["bias_cat_1"].values))[0]
                bias_cat_2 = list(set(model_df["bias_cat_2"].values))[0]
                # make confusion matrix and also a zero'd along the diagonal confusion matrix for the agreements (so colours easier to see)
                if args.include_gold:
                    # confusion matrix is the diff between the matrices for privileged and not privileged
                    cm1 = metrics.confusion_matrix(model_df["gold_label_int"].values,
                                                   model_df["label_1"].values,
                                                   labels=labels)
                    cm2 = metrics.confusion_matrix(model_df["gold_label_int"].values,
                                                   model_df["label_2"].values,
                                                   labels=labels)
                    cm = cm1 - cm2
                    ylabel = "gold"
                    xlabel = "predicted"
                    # make new cmap to be sure to anchor it to white as zero
                    cmap = plt.cm.bwr  # diverging cmap since looking at a diff
                    vmin, vmax = np.min(cm), np.max(cm)
                    cmap = shifted_colormap(cmap, midpoint=get_cmap_midpoint(vmin, vmax), name=f'{args.lang}_{bt}_{model_type}_cmap')
                    
                else:
                    cmap = 'Blues'
                    ylabel = bias_cat_1
                    xlabel = bias_cat_2
                    cm = metrics.confusion_matrix(model_df["label_1"].values, model_df["label_2"].values,
                                              labels=labels)
                    #include also a mod version so that the non-agreements are informative to set the min and max
                    cm_mod = cm.copy()
                    cm_max = cm.copy()
                    all_max = np.max(cm)
                    for i in range(len(labels)):
                        cm_mod[i][i] = 0
                        cm_max[i][i] = all_max # so that ensure it is black
                    # this is for anchoring the colour map to useful values
                    vmin, vmax = np.min(cm_mod), np.max(cm_mod)
                    my_cmap = plt.get_cmap(cmap).copy()
                    my_cmap.set_over('black')

                    if args.polarity:
                        labels = ["negative", "neutral", "positive"]
                    myplot = sns.heatmap(cm_max, xticklabels=labels, yticklabels=labels, cmap=my_cmap, vmin=vmin, vmax=vmax)
                    outfile_mod  = os.path.join(args.output_dir,
                                                         f"{args.lang}_{bt}_{model_type}_cap.pdf")
                    plt.ylabel(ylabel)
                    plt.xlabel(xlabel)
                    plt.savefig(outfile_mod)
                    plt.clf()

                if args.polarity:
                    labels = ["negative", "neutral", "positive"]
                myplot = sns.heatmap(cm, xticklabels=labels, yticklabels=labels, cmap=cmap)
                outfile = os.path.join(args.output_dir, f"{args.lang}_{bt}_{model_type}.pdf")
                plt.ylabel(ylabel)
                plt.xlabel(xlabel)
                plt.savefig(outfile)
                plt.clf()

        else:
            # set output filename
            outfile = os.path.join(args.output_dir, f"all_models_{args.lang}_{bt}.pdf")
            if args.plot_type == "bubble":
                myplot = ggplot(this_df, aes(x="model_type", y="performance_gap")) + geom_count(color=colour)
                myplot.save(outfile)
            else:
                if args.plot_type == "violin":
                    myplot = sns.violinplot(data=this_df, x="model_type", y="performance_gap", color=colour, order=type_order)
                elif args.plot_type == 'errbars':
                    #ipdb.set_trace()
                    myplot = sns.lineplot(data=this_df, x="model_type", y="performance_gap",
                                           color=colour, linestyle='',
                                           err_style='bars', marker='o')
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
                if args.plot_type == "errbars" or args.plot_type == "scatter":
                    myplot.axhspan(xspan,-xspan,alpha=0.2)
                    #myplot.axhline(0.12, linestyle="--", color="gray")
                    #myplot.axhline(-0.12, linestyle="--", color="gray")
                myplot.axhline(0.0, linestyle=":", color="gray")
                plt.xticks(rotation=90)
                plt.ylim(*y_axis)
                plt.subplots_adjust(bottom=0.28)
                plt.savefig(outfile)
                plt.clf()

