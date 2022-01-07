"""
Script that reads in results csvs by emotion and adds binary performance data columns:
gold_label, pred_label.

Note that this can only be done with the full_output (including labels) data

Then uses them to calculate performance metrics.
"""
import argparse
import os

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from utils.data_utils import get_label_from_emotion, convert_to_polarity


def setup_argparse():
    p = argparse.ArgumentParser()
    p.add_argument('-r', dest='results_dir', default='', help='')
    p.add_argument('--remove_neutral', action='store_true', help='turn labels into binary instead of ternary')
    return p.parse_args()


if __name__ == "__main__":
    args = setup_argparse()

    labels = ["label_1", "label_2"]
    cats = ["bias_cat_1", "bias_cat_2"]
    with os.scandir(args.results_dir) as source_dir:
        files = [f.name for f in source_dir if f.is_file() and f.name.endswith("emotion.csv")]

    # for writing out a dataframe
    perf_columns = ["precision", "recall", "f1", "support", "acc", "bias_cat", "bias_type", "steps"]
    for file_name in files:
        df = pd.read_csv(os.path.join(args.results_dir, file_name))
        print(file_name)
        # convert labels to their polarity
        for label in labels:
            df[label] = df[label].apply(convert_to_polarity)
        # add column for gold label
        gold_labels = df['emotion'].apply(get_label_from_emotion)
        df['gold_label'] = gold_labels

        # Now print out stats and write out also
        perf_data = []
        bias_types = set(df["bias_type"].values)
        for bias_type in bias_types: # these will be the type of categories
            print(bias_type)
            mask = df["bias_type"] == bias_type
            df_by_bias = df[mask]
            for model in sorted(list(set(df["steps"].values))):
                print(model)
                mask = df_by_bias['steps'] == model
                df_by_model = df_by_bias[mask]
                if args.remove_neutral:
                    neutral_mask = df_by_model["gold_label"] == "neutral"
                    df_by_model = df_by_model[~neutral_mask]
                gold = df_by_model["gold_label"].values
                for subgroup_label, subgroup_cat in zip(labels, cats):
                    subgroup_cat_val = list(set(df_by_model[subgroup_cat].values))[0] # should be only one
                    pred = df_by_model[subgroup_label].values
                    precision, recall, f1, support = precision_recall_fscore_support(gold, pred, average="macro")
                    acc = accuracy_score(gold, pred)
                    print(f"Precision: {precision}\n Recall: {recall}\n F1: {f1}\n Acc: {acc}\n")
                    perf_data.append([precision, recall, f1, support, acc, subgroup_cat_val, bias_type, model])

        prev_filename, ext = os.path.splitext(file_name)
        new_filename = prev_filename + "_performance_binary" + ext if args.remove_neutral else prev_filename + "_performance" + ext
        new_filepath = os.path.join(args.results_dir, new_filename)
        perf_df = pd.DataFrame(perf_data, columns=perf_columns)
        perf_df.to_csv(new_filepath, index=False)





