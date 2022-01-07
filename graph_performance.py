import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    ### Now graph the performance
    type2pattern = {
        "mono": "/home/ec2-user/SageMaker/efs/sgt/results/{}/full_output/{}_ensemble_all_data_emotion_performance.csv",
        "multi": "/home/ec2-user/SageMaker/efs/sgt/results/{}/full_output/multi+en_{}_ensemble_all_data_emotion_performance.csv"
    }
    
    output_path_f1 = "/home/ec2-user/SageMaker/efs/sgt/results/performance_graphs/{}_{}_f1.png"
    output_path_acc = "/home/ec2-user/SageMaker/efs/sgt/results/performance_graphs/{}_{}_acc.png"
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