import argparse
import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils.model_utils import lang2convergence, compressed2convergence, multi_convergence, \
    compressed_multi_convergence, balanced_multi_convergence

def setup_argparse():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', action='store_true')
    p.add_argument('--compressed', action='store_true')
    p.add_argument('--balanced', action='store_true')
    p.add_argument('-o', dest='output_dir', default='analysis/plot_all/', help='output dir')
    
    return p.parse_args()
    
if __name__ == "__main__":
    args = setup_argparse()

    baseline_pattern = "results/baseline/{}/{}_ensemble.csv"
    pattern = "results/{}/{}_ensemble.csv"
    multi_pattern = "results/{}/multi+en_{}_ensemble.csv"

    if args.compressed:
        pattern = "results/compressed/{}/{}_ensemble.csv"
        multi_pattern = "results/compressed/{}/multi+en_{}_ensemble.csv"

        multi_convergence = compressed_multi_convergence

    if args.balanced:
        multi_pattern = "results/balanced/{}/multi+en_{}_ensemble.csv"
        multi_convergence = balanced_multi_convergence


    master_df = pd.DataFrame()
    #y_axis = (-0.2, 0.4) # min and max of the output
    y_axis = (-0.4, 0.8)

    convergence_dict = compressed2convergence if args.compressed else lang2convergence

    if not args.baseline:
        for lang, steps in convergence_dict.items():
            print(lang)
            df = pd.read_csv(pattern.format(lang, lang))
            #print(df)
            if lang == "en_scrubbed":
                df["lang"] = "en_s"
            mask = df["steps"] == convergence_dict[lang]
            df = df[mask]
            master_df = master_df.append(df)
            #myplot = sns.scatterplot(data=df, x="steps", y="performance_gap")

        mask = master_df["bias_type"] == "rank"
        master_df = master_df[~mask]

        # add column
        master_df["mono_multi"] = "mono"

        master_df_multi = pd.DataFrame()
        for lang, steps in convergence_dict.items():
            if lang == "en_scrubbed":
                continue
            print(lang)
            df = pd.read_csv(multi_pattern.format(lang, lang))
            mask = df["steps"] == multi_convergence
            df = df[mask]
            #print(set(df['lang'].values))
            master_df_multi = master_df_multi.append(df)

        mask = master_df_multi["bias_type"] == "rank"
        master_df_multi = master_df_multi[~mask]

        master_df_multi["mono_multi"] = "multi"

        full_data = master_df.append(master_df_multi)
        for m in ["mono", "multi"]:
            mask = full_data["mono_multi"] == m
            this_df = full_data[mask]
            order = ["en", "de", "es", "ja", "zh"]  # cannot order a scatterplot it seems...
            if m == "mono":
                marker = "o"
                order.insert(1, "en_scrubbed")
            else:
                marker = "D"
            significance_mask = this_df["statistical_significance"] > 0.05
            sig_df = this_df[significance_mask]
            myplot = sns.scatterplot(data=this_df, x="lang", y="performance_gap", hue="bias_type", marker=marker)
            myplot = sns.scatterplot(data=sig_df, x="lang", y="performance_gap", color="black", marker="x", s=100)
            myplot.set(ylabel=None, yticklabels=[])
            myplot.tick_params(left=False)
            myplot.axhline(0.0, linestyle=":", color="gray")
            plt.ylim(*y_axis)
            plt.savefig(os.path.join(args.output_dir, f"all_data_{m}.pdf"))
            plt.clf()
    
    else:
        for lang in convergence_dict.keys():
            df = pd.read_csv(baseline_pattern.format(lang, lang))
            #print(df)
            if lang == "en_scrubbed":
                df["lang"] = "en_s"
            master_df = master_df.append(df)
            #myplot = sns.scatterplot(data=df, x="steps", y="performance_gap")

        mask = master_df["bias_type"] == "rank"
        master_df = master_df[~mask]
        marker = "o"
        myplot = sns.scatterplot(data=master_df, x="lang", y="performance_gap", hue="bias_type", marker=marker)
        significance_mask = master_df["statistical_significance"] > 0.05
        sig_df = master_df[significance_mask]
        myplot = sns.scatterplot(data=sig_df, x="lang", y="performance_gap", color="black", marker="x", s=100)
        myplot.axhline(0.0, linestyle=":", color="gray")
        plt.ylim(*y_axis)
        plt.savefig(os.path.join(args.output_dir,"all_data_baseline.pdf"))
        plt.clf()

            