import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def setup_argparse():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', action='store_true')
    return p.parse_args()


if __name__ == "__main__":
    args = setup_argparse()
    ### Now graph the performance
    type2pattern = {
        "mono": "results/{}/full_output/{}_ensemble_all_data_emotion_performance.csv",
        "multi": "results/{}/full_output/multi+en_{}_ensemble_all_data_emotion_performance.csv"
    }
    patterns = [] if args.baseline else type2pattern.keys()

    
    output_path_f1 = "analysis/performance_graphs/{}_{}_f1.png"
    output_path_acc = "analysis/performance_graphs/{}_{}_acc.png"
    for lang in ["en", "zh", "ja", "es", "de"]:
        print(f"working on {lang}")
        for pattern in type2pattern.keys():
            df = pd.read_csv(type2pattern[pattern].format(lang,lang))
            myplot = sns.scatterplot(data=df, x="steps", y="f1", style="bias_cat")

            plt.savefig(output_path_f1.format(pattern, lang))
            plt.clf()
            
            myplot = sns.scatterplot(data=df, x="steps", y="acc", style="bias_cat")

            plt.savefig(output_path_acc.format(pattern, lang))
            plt.clf()