import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

lang2convergence = {
    "de": 35720, #36720,
    "es": 35720, #35190, #OR 41310, 0.61
    "ja": 42864, #44370,
    "zh": 35720, #35190,
    "en": 28576, #27050,
    "en_scrubbed": 25000, #26520
    #"multi": 100644, # 78540,
}
baseline_pattern = "/home/ec2-user/SageMaker/efs/sgt/results/baseline/{}/{}_ensemble.csv"

pattern = "/home/ec2-user/SageMaker/efs/sgt/results/{}/{}_ensemble.csv"
multi_pattern = "/home/ec2-user/SageMaker/efs/sgt/results/{}/multi+en_{}_ensemble.csv"

    
def setup_argparse():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', action='store_true')
    
    return p.parse_args()
    
if __name__ == "__main__":
    args = setup_argparse()
    
    master_df = pd.DataFrame()
    #y_axis = (-0.2, 0.4) # min and max of the output
    y_axis = (-0.4, 0.8)
    
    if not args.baseline:
        for lang, steps in lang2convergence.items():
            print(lang)
            df = pd.read_csv(pattern.format(lang, lang))
            #print(df)
            if lang == "en_scrubbed":
                df["lang"] = "en_s"
            mask = df["steps"] == lang2convergence[lang]
            df = df[mask]
            master_df = master_df.append(df)
            #myplot = sns.scatterplot(data=df, x="steps", y="performance_gap")

        mask = master_df["bias_type"] == "rank"
        master_df = master_df[~mask]


        # add column
        master_df["mono_multi"] = "mono"

        master_df_multi = pd.DataFrame()
        for lang, steps in lang2convergence.items():
            if lang == "en_scrubbed":
                continue
            print(lang)
            df = pd.read_csv(multi_pattern.format(lang, lang))
            mask = df["steps"] == 100644
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
            plt.savefig(f"all_data_{m}.pdf")
            plt.clf()
    
    else:
        for lang in lang2convergence.keys():
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
        plt.savefig("all_data_baseline.pdf")
        plt.clf()

            