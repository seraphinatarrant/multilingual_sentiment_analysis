import argparse
import os

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# from utils.data_utils import BY_SEED_HEADERS, EMO_HEADERS, CSV_BASE
from evaluation.create_eval_set import lang2bias, EMOTION_CATEGORIES


def setup_argparse():
    p = argparse.ArgumentParser()
    p.add_argument('-r', dest='results_dir', default='results/')
    p.add_argument('-o', dest='output_dir', default='results/graphs/')
    p.add_argument('--langs', nargs='+', default=['ja'], help='')
    p.add_argument('--seeds', nargs='+', default=[0, 5, 26, 42, 63, "ensemble"], help='')
    p.add_argument('--by_emotion', action='store_true')
    p.add_argument('--bias_cat', nargs='+', help='if looking a specific bias category rather than all for a language')
    p.add_argument('--include_multi', action='store_true')
    p.add_argument('--binarise_stat_sig', action='store_true', help='convert stat sig to binary based on alpha')
    p.add_argument('--alpha', type=float, default=0.05, help='alpha to use for binary stat_sig')
    p.add_argument('--together', action='store_true')  ## modify stuff below to also graph them together
    p.add_argument('--plot_type', choices=["scatter", "box", "violin", "pointplot"], default="scatter")

    return p.parse_args()


def graph_results(dataframe, output_path, y_axis=None, plot_type="scatter"):
    if plot_type == "scatter":
        palette = { #TODO is this fine if not binarise?
            0: "grey",
            1: "mediumblue"
        }
        myplot = sns.scatterplot(data=dataframe, x="steps", y="performance_gap", hue="statistical_significance", palette=palette)
    if plot_type == "pointplot":
        myplot = sns.pointplot(data=dataframe, x="steps", y="performance_gap")
    if plot_type == "box":
        myplot = sns.boxplot(data=dataframe, x="steps", y="performance_gap")
    if plot_type == "violin":
        myplot = sns.violinplot(data=dataframe, x="steps", y="label", hue="bias_cat_name", split=True)
    # print(lang)
    # print(bias, seed)
    if y_axis and plot_type == "scatter":  # only want to fix for scatter, since otherwise scale to outliers
        y_min, y_max = y_axis
        plt.ylim(y_min * 1.2, y_max * 1.2)
    # plt.show()
    plt.xticks(rotation=90)
    plt.savefig(output_path)
    plt.clf()


def get_global_min_max(csv_paths, bias_cat, lang, seeds, column_name):
    all_csvs = []
    for csv_path in csv_paths:
        all_csvs.extend([csv_path.format(lang, seed) for seed in seeds])
    # print(all_csvs)
    curr_min, curr_max = 1000, -1000
    for c in all_csvs:
        this_csv = pd.read_csv(c)
        mask = this_csv['bias_type'].str.contains(bias_cat)
        dataframe = this_csv[mask]
        this_min, this_max = dataframe[column_name].min(), dataframe[column_name].max()
        curr_min = this_min if this_min < curr_min else curr_min
        curr_max = this_max if this_max > curr_max else curr_max
    return curr_min, curr_max


if __name__ == "__main__":
    args = setup_argparse()

    if args.plot_type == "scatter":
        csv_mono_path = "{}_{}_emotion.csv" if args.by_emotion else "{}_{}.csv"
    else:
        csv_mono_path = "{}_{}_all_data_emotion.csv" if args.by_emotion else "{}_{}_all_data.csv"
    csv_multi_path = "multi+en_" + csv_mono_path
    for lang in args.langs:
        bias_categories = args.bias_cat if args.bias_cat else lang2bias[lang]
        # lang = f"{lang}_overall" if args.seeds[0] == "all" else lang # means uses all seeds
        all_csvs = [csv_mono_path, csv_multi_path] if args.include_multi else [csv_mono_path]
        for bias in bias_categories:
            y_min_max = get_global_min_max([os.path.join(args.results_dir, c) for c in all_csvs],
                                           bias,
                                           lang,
                                           args.seeds,
                                           "performance_gap")

            for seed in args.seeds:
                for csv_path in all_csvs:
                    results = os.path.join(args.results_dir, csv_path.format(lang, seed))
                    results = pd.read_csv(results)
                    # convert stat sig numbers to binary if necessary
                    if args.binarise_stat_sig:
                        results['statistical_significance'] = results['statistical_significance'].map(
                            lambda value: 1 if value < args.alpha else 0)
                    mask = results['bias_type'].str.contains(bias)
                    dataframe = results[mask]
                    if args.by_emotion:
                        output_filename = os.path.join(args.output_dir,
                                                       f'{lang}_{seed}_{bias}_{emotion}_{args.plot_type}.png') if csv_path == csv_mono_path else os.path.join(
                            args.output_dir, f'multi+en_{lang}_{seed}_{bias}_{emotion}_{args.plot_type}.png')
                        for emotion in set(dataframe["emotion"].tolist()):
                            # print(emotion)
                            emotion_mask = dataframe['emotion'].str.contains(emotion)
                            emotion_df = dataframe[mask]
                            graph_results(
                                emotion_df,
                                output_filename,
                                y_min_max, plot_type=args.plot_type)
                    else:
                        output_filename = os.path.join(args.output_dir,
                                                       f'{lang}_{seed}_{bias}_{args.plot_type}.png') if csv_path == csv_mono_path else os.path.join(
                            args.output_dir,
                            f'multi+en_{lang}_{seed}_{bias}_{args.plot_type}.png')
                        graph_results(dataframe, output_filename, y_min_max, plot_type=args.plot_type)
