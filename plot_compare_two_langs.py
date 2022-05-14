import argparse
import os
import sys
import ipdb

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plotnine import ggplot, aes, geom_count
from sklearn import metrics

from utils.model_utils import lang2convergence, compressed2convergence, multi_convergence, \
    compressed_multi_convergence, balanced_multi_convergence, compressed_balanced_multi_convergence, mono_multi2convergence

from evaluation.create_eval_set import lang2bias

def get_convergence_by_type(model_type, lang):
    """
    multilingual models are zero shot to training data so the convergence numbers are the same
    regardless of language applied to. Monolingual models are different per language so they have
    a dict to query convergence steps by language
    """
    if model_type == "mono":
        steps = lang2convergence[lang]
    elif model_type == "mono_c":
        steps = compressed2convergence[lang]
    else:
        sys.exit("Not a valid model type to check for convergence: {}".format(model_type))

    return steps


def setup_argparse():
    p = argparse.ArgumentParser()
    p.add_argument('-l', '--lang', dest='lang', default="en")
    p.add_argument('--compare_lang', default="en_scrubbed", help='language to compare against the first')
    p.add_argument('-o', dest='output_dir', default='analysis/compare_langs/', help='output dir')
    #p.add_argument('-pt', '--plot_type', choices=['heatmap', 'scatter', 'bubble', 'violin', "errbars"],
    #               default="errbars")
    p.add_argument('--polarity', action='store_true', help='use results that have been preconverted to polarity')
    #p.add_argument('--include_gold', action='store_true', help='if doing polarity, can also include '
    #                                                           'gold results, currently only works with heatmap')

    return p.parse_args()


if __name__ == "__main__":
    args = setup_argparse()

    print(args)

    type_order = ["baseline", "mono", "mono_c"]

    type2filepattern = {
        "baseline": "results/baseline/{}/{}_ensemble.csv",
        "mono": "results/{}/{}_ensemble.csv",
        "mono_c": "results/compressed/{}/{}_ensemble.csv",       
    }

    insert = "full_output"
    file_insert = "_all_data_emotion_polarity" if args.polarity else "_all_data"
    for key, val in type2filepattern.items():
        _path, _filename = os.path.split(val)
        _file, _ext = os.path.splitext(_filename)
        new_filename = _file + file_insert + _ext
        new_val = os.path.join(_path, insert, new_filename)
        type2filepattern[key] = new_val

    num_models = len(type_order)

    # This all just makes the correct dataframes
    master_df = pd.DataFrame()
    print("Gathering dataframes...")
    for lang in [args.lang, args.compare_lang]:
        for model_type, file_pattern in type2filepattern.items():
            print(model_type)
            try:
                infile = file_pattern.format(lang, lang) if model_type != "multi_on_mono" else file_pattern.format(lang, lang, lang)
                df = pd.read_csv(infile)
            except:
                print("Couldn't read in file, may not exist: {}".format(infile), file=sys.stderr)
                continue
            if lang == "en_scrubbed":
                df["lang"] = "en_scrubbed"
            if model_type != "baseline":
                convergence_steps = get_convergence_by_type(model_type, lang)
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

        # set output filename
        outfile = os.path.join(args.output_dir, f"compare_langs_{args.lang}_{args.compare_lang}_{bt}")
        # make errbar plot
        y_axis = (-0.15, 0.25)
        myplot = sns.lineplot(data=this_df, x="model_type", y="performance_gap", style="lang",
                              color=colour, linestyle='',
                              err_style='bars') #, marker='o') Can manually set markers if it doesn't work well auto
        myplot.axhspan(0.1, -0.1, alpha=0.2)
        myplot.axhline(0.0, linestyle=":", color="gray")
        plt.xticks(rotation=90)
        plt.ylim(*y_axis)
        #plt.subplots_adjust(bottom=0.28)
        plt.savefig(outfile+"_errbars.pdf")
        plt.clf()

        # make violin plot
        y_axis = (-4, 4)
        myplot = sns.violinplot(data=this_df, x="model_type", y="performance_gap", hue="lang",
                                        color=colour, order=type_order)

        myplot.axhline(0.0, linestyle=":", color="gray")
        plt.xticks(rotation=90)
        plt.ylim(*y_axis)
        # plt.subplots_adjust(bottom=0.28)
        plt.savefig(outfile + "_violin.pdf")
        plt.clf()

        # do confusion matrices
        # if doing a confusion matrix heatmap, then need to also do a separate graph for each model
        labels = range(1, 6)
        print(f"Difference of {args.lang} and {args.compare_lang} (first minus second)")
        for model_type in type_order:
            #ipdb.set_trace()
            mask = this_df["model_type"] == model_type
            model_df = this_df[mask]
            mask_lang1 = model_df["lang"] == args.lang
            mask_lang2 = model_df["lang"] == args.compare_lang
            lang1_df = model_df[mask_lang1]
            lang2_df = model_df[mask_lang2]
            # confusion matrix is the diff between the matrices for privileged and not privileged
            cm1 = metrics.confusion_matrix(lang1_df["label_1"].values,
                                               lang1_df["label_2"].values,
                                               labels=labels)
            cm2 = metrics.confusion_matrix(lang2_df["label_1"].values,
                                           lang2_df["label_2"].values,
                                           labels=labels)
            cm = cm1 - cm2
            myplot = sns.heatmap(cm, xticklabels=labels, yticklabels=labels)
            bias_cat_1 = list(set(model_df["bias_cat_1"].values))[0]
            bias_cat_2 = list(set(model_df["bias_cat_2"].values))[0]
            plt.ylabel(bias_cat_1)
            plt.xlabel(bias_cat_2)
            outfile = os.path.join(args.output_dir, f"compare_langs_{args.lang}_{args.compare_lang}_{model_type}.pdf")
            plt.savefig(outfile)
            plt.clf()
