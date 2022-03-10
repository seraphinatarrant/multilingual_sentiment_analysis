import argparse
import csv
import math
import sys
import os
import typing
import yaml
import json
from collections import defaultdict, Counter

from scipy.stats import ttest_rel
import numpy as np

from utils.model_utils import load_model_and_tokenizer
from utils.data_utils import BY_SEED_HEADERS, OVERALL_HEADERS, EMO_ADDITION, LABEL_HEADERS, CSV_BASE
from utils.data_utils import load_corpus
from evaluation.create_eval_set import lang2bias
from contrastive_evaluation import eval_model_on_corpus
from baseline_classifier import eval_baseline_on_corpus, load_saved


def get_info_from_top_dir(dir_path):
    all_info = dir_path.split("_")
    lang, epochs, seed = all_info[0], all_info[1], all_info[2]

    return lang, epochs, seed


def setup_argparse():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='models/experiment_ja.yaml', help='')
    p.add_argument('--output', default='/home/ec2-user/SageMaker/efs/sgt/results/')
    p.add_argument('--average', choices=["mean", "ensemble"], default="mean")
    p.add_argument('--baseline', action='store_true', help='use baseline models from scikit learn')
    return p.parse_args()


def get_info_from_checkpoint(dir_path):
    _, steps = dir_path.split("-")
    return steps

# sort out splitting off initial models directory?


def get_bias_stats(bias_cat_1_name: str, bias_cat_2_name: str, bias2results: dict):
    statistic, p = ttest_rel(bias2results[bias_cat_1_name], bias2results[bias_cat_2_name])
    bias_cat_1_mean, bias_cat_2_mean = np.mean(bias2results[bias_cat_1_name]), \
                                       np.mean(bias2results[bias_cat_2_name])
    performance_values = np.subtract(bias2results[bias_cat_1_name], bias2results[bias_cat_2_name])
    performance_gap = bias_cat_1_mean - bias_cat_2_mean  # assumes category 1 is privileged, otherwise sign is wrong

    return bias_cat_1_mean, bias_cat_2_mean, performance_gap, performance_values, p


def majority_vote(x):
    c = Counter(x)
    # check what the highest frequency number is, and if > half of len(x) (in our case, 3+) then just take highest
    most_freq_label, highest_count = c.most_common(1)[0] # second in tuple is count
    if highest_count > len(x)//2:
        return most_freq_label, False #no disagreement so return 0 for that
    else:
        # TODO sort out how to find out what example this was -- helpful that the order of the corpus is fixed
        if len(c) == len(x): # all one vote, return middle
            print("disagreement, all different votes, returning middle", file=sys.stderr)
            middle_idx = math.ceil(len(x) / 2)
            return sorted(x)[middle_idx], True
        else: # one vote has one more, or 2,2,1
            if len(c) == len(x)-1:
                print("disagreement, only one vote has more, using it", file=sys.stderr)
            else:
                second_freq, second_count = c.most_common(2)[1]
                print(f"disagreement, two labels are tied, arbitrarily picking one, "
                      f"labels: {most_freq_label}, {second_freq} ", file=sys.stderr)
            # in the 2,1,1,1,1 case, technically no tie, but want to log as disagreement
            # in the 2,2,1 case, just arbitrarily break ties (deterministically, as Counter will return the one it saw first I believe)
            return most_freq_label, True


def validate_bias_types(bias_cats, results_dict):
    intersect = set(bias_cats) & results_dict.keys()
    if intersect != set(bias_cats):
        print(f"All bias types are not in results.\n Tried Keys: {bias_cats}\n "
                  f"Keys found:{results_dict.keys()}\n", file=sys.stderr)
        return False
    else:
        return True


def ensemble(bias_categories, all_results_by_seed):
    all_results = defaultdict(lambda: defaultdict(list))
    for bias_cat in bias_categories:
        print(bias_cat, file=sys.stderr)
        total_disagreements = 0
        max_samples = 0
        for bcs in bias_categories[bias_cat]:
            for steps in sorted(all_results_by_seed.keys()):
                print("Results for models at {} steps:".format(steps), file=sys.stderr)
                disagreements, disagreement_indices = 0, []

                seed2results = all_results_by_seed[steps][bcs]
                if not seed2results:
                    continue
                new_results = []
                # now majority vote
                for index, scores in enumerate(zip(*list(seed2results.values()))):
                    winner, disagreement = majority_vote(scores)
                    if disagreement:
                        disagreements += 1
                        disagreement_indices.append(index)

                    new_results.append(winner)
                all_results[steps][bcs] = new_results
                print("All disagreement indices: {}".format(disagreement_indices), file=sys.stderr)
                print("{} disagreements between models of {} samples ({:.2f}%)".format(disagreements,
                                                                                       index + 1,
                                                                                       disagreements / (
                                                                                                   index + 1) * 100),
                      file=sys.stderr)
                print("Done with {} steps".format(steps), file=sys.stderr)
                total_disagreements += disagreements
                max_samples += index + 1
        if max_samples > 0:
            print("{} total disagreements across all checkpoints "
                  "for {} total samples ({:.2f}%)".format(total_disagreements, max_samples,
                                                          (total_disagreements / max_samples) * 100), file=sys.stderr)
    return all_results


if __name__ == "__main__":
    args = setup_argparse()

    config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    model_lang, test_lang = config['model_lang'], config['test_lang']
    model_dirs = config['model_dirs'] # these will have multiple dirs inside them for loading

    # load test corpus
    test_corpus = load_corpus(test_lang)

    bias_categories = lang2bias[test_lang]

    # topdir is one seed, under which there are many models at different checkpoints
    topdir2models = {}
    for model_dir in model_dirs:
        with os.scandir(model_dir) as source_dir:
            topdir2models[model_dir] = sorted([d.name for d in source_dir if d.is_dir() and d.name.startswith('checkpoint')])

    all_results, all_results_by_emotion = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(dict)) # bias
    #in case of ensembling
    all_results_by_seed, all_results_by_emotion_by_seed = defaultdict(
        lambda: defaultdict(dict)), defaultdict(lambda: defaultdict(dict))  # bias
    #avg_results, avg_results_by_emotion = defaultdict(list), defaultdict(lambda: defaultdict(list))
    # for each model in each seed directory, write its own csv. But also store results to write an averaged csv at the end
    for topdir, models in topdir2models.items():
        print("Processing {} models in {}".format(len(models), topdir), file=sys.stderr)
        dirname = [s for s in topdir.split(os.path.sep) if s][-1]  #selects the last component of path and deals with empty strings
        lang, _, seed = get_info_from_top_dir(dirname)
        csv_base = CSV_BASE.format(lang, seed) if model_lang == test_lang else CSV_BASE.format(f'{lang}_{test_lang}', seed)
        csv_name, csv_emo_name = os.path.join(args.output, csv_base + ".csv"), os.path.join(args.output, csv_base + "_emotion.csv")
        # put full output in different dir
        full_output_dir = os.path.join(args.output, "full_output")
        if not os.path.exists(full_output_dir):
            os.makedirs(full_output_dir)
        csv_labels_name, csv_labels_emo_name = os.path.join(full_output_dir, csv_base + "_all_data.csv"), os.path.join(full_output_dir, csv_base + "_all_data_emotion.csv")
        with open(csv_name, "w", newline='') as csvout, open(csv_emo_name, "w", newline='') as csvemo,\
            open(csv_labels_name, "w", newline='') as csv_all, open(csv_labels_emo_name, "w", newline='') as csv_all_emo:
            csv_writer, csv_emo_writer = csv.writer(csvout), csv.writer(csvemo)
            csv_writer_all, csv_writer_all_emo = csv.writer(csv_all), csv.writer(csv_all_emo)

            csv_writer.writerow(BY_SEED_HEADERS)
            csv_emo_writer.writerow(BY_SEED_HEADERS + EMO_ADDITION)
            csv_writer_all.writerow(LABEL_HEADERS)
            csv_writer_all_emo.writerow(LABEL_HEADERS + EMO_ADDITION)

            for model in models:
                if args.baseline:
                    steps = 0
                    classifier_pipeline = load_saved(os.path.join(topdir, model))
                    bias2results, emo2bias2results, bias2performance, results_performance = eval_baseline_on_corpus(
                        classifier_pipeline, test_corpus
                    )
                else:
                    steps = get_info_from_checkpoint(model)
                    try:
                        classifier_pipeline = load_model_and_tokenizer(os.path.join(topdir, model),
                                                                   from_path=True, return_pipeline=True)
                    except:
                        print("Failed to load: {}".format(os.path.join(topdir, model)), file=sys.stderr)

                    # bias2results returns e.g. "male: [1,3,3,5]", emo2bias2results returns eg. 'anger': 'male': [1,3,3,5]
                    bias2results, emo2bias2results = eval_model_on_corpus(classifier_pipeline, test_corpus)

                # save for later for overall stats
                for b, r in bias2results.items():
                    all_results[steps][b].extend(r)

                    # This is necessary for ensembling
                    all_results_by_seed[steps][b][seed] = r

                for emo in emo2bias2results:
                    for b, r in emo2bias2results[emo].items():
                        # standard
                        if b in all_results_by_emotion[emo][steps]:
                            all_results_by_emotion[emo][steps][b].extend(r)
                        else:
                            all_results_by_emotion[emo][steps][b] = r
                        # by seed
                        if b in all_results_by_emotion_by_seed[emo][steps]:
                             all_results_by_emotion_by_seed[emo][steps][b].update({seed:  r})
                        else:
                            all_results_by_emotion_by_seed[emo][steps].update({b: {seed:  r}})

                for bias_cat in bias_categories: # bias_cat is high_level
                    bias_cat_1_name, bias_cat_2_name = bias_categories[bias_cat]  # there should only be 2 since tests are paired

                    # overall bias
                    bias_cat_1_mean, bias_cat_2_mean, performance_gap, performance_values, p = get_bias_stats(bias_cat_1_name, bias_cat_2_name, bias2results)

                    # write specific labels
                    # ["label_1", "label_2", "performance_gap", "bias_cat_1", "bias_cat_2", "bias_type", "steps", "lang"] (and + emo for emotion one)
                    results_1, results_2 = bias2results[bias_cat_1_name], bias2results[bias_cat_2_name]
                    for idx, vals in enumerate(list(zip(results_1, results_2))): # val is the performance_gap
                        csv_writer_all.writerow([vals[0], vals[1], performance_values[idx], bias_cat_1_name,
                                                 bias_cat_2_name, bias_cat, steps, test_lang])
                        for emo in emo2bias2results:
                            results_1, results_2 = emo2bias2results[emo][bias_cat_1_name], emo2bias2results[emo][bias_cat_2_name]
                            for idx, vals in enumerate(list(zip(results_1, results_2))):  # val is the performance_gap
                                csv_writer_all_emo.writerow([vals[0], vals[1], performance_values[idx], bias_cat_1_name,
                                                         bias_cat_2_name, bias_cat, steps, test_lang, emo])


                    # write overall
                    csv_writer.writerow([bias_cat_1_mean, bias_cat_2_mean, performance_gap, p, bias_cat_1_name,
                                        bias_cat_2_name, bias_cat, steps, test_lang, topdir])

                    for emotion in emo2bias2results:
                        bias_cat_1_mean, bias_cat_2_mean, performance_gap, performance_values, p = get_bias_stats(bias_cat_1_name, bias_cat_2_name, emo2bias2results[emotion])
                        csv_emo_writer.writerow([bias_cat_1_mean, bias_cat_2_mean, performance_gap, p, bias_cat_1_name,
                                            bias_cat_2_name, bias_cat, steps, test_lang, topdir, emotion])


    # by seed results have now been finished,
    # now average and then write that out for overall
    for steps in all_results:
        for bias, res in all_results[steps].items():
            # average the nested list that is all_results[bias][steps]
            avg_results = np.mean(res, axis=0)  # makes it do the average columnwise down each row (and each row is one seed)
            all_results[steps][bias] = avg_results
    for emo in all_results_by_emotion:
        for steps in all_results_by_emotion[emo]:
            for bias, res in all_results_by_emotion[emo][steps].items():
                avg_results_by_emotion = np.mean(res, axis=0)
                all_results_by_emotion[emo][steps][bias] = avg_results_by_emotion

    if args.average == "mean":
        averaging_method = "overall"
    else:
        averaging_method = "ensemble"
        # if using ensemble, convert to majority vote from steps: bias: seed: results
        all_results = ensemble(bias_categories, all_results_by_seed)  # overwrites previous all_results
        all_emotions = emo2bias2results.keys()
        print("Ensembling by emotion", file=sys.stderr)
        for emo in all_emotions:
            print(emo, file=sys.stderr)
            all_results_by_emotion[emo] = ensemble(bias_categories, all_results_by_emotion_by_seed[emo])


    csv_base = CSV_BASE.format(lang, averaging_method) if model_lang == test_lang else CSV_BASE.format(f'{lang}_{test_lang}', averaging_method)
    csv_name, csv_emo_name = os.path.join(args.output, csv_base + ".csv"), os.path.join(args.output, csv_base + "_emotion.csv")
    csv_labels_name, csv_labels_emo_name = os.path.join(full_output_dir, csv_base + "_all_data.csv"), os.path.join(full_output_dir, csv_base + "_all_data_emotion.csv")
    with open(csv_name, "w", newline='') as csvout, open(csv_emo_name, "w", newline='') as csvemo, \
            open(csv_labels_name, "w", newline='') as csv_all, open(csv_labels_emo_name, "w", newline='') as csv_all_emo:
        csv_writer, csv_emo_writer = csv.writer(csvout), csv.writer(csvemo)
        csv_writer_all, csv_writer_all_emo = csv.writer(csv_all), csv.writer(csv_all_emo)
        csv_writer.writerow(OVERALL_HEADERS)
        csv_emo_writer.writerow(OVERALL_HEADERS + EMO_ADDITION)
        csv_writer_all.writerow(LABEL_HEADERS)
        csv_writer_all_emo.writerow(LABEL_HEADERS + EMO_ADDITION)

        for bias_cat in bias_categories:  # bias_cat is high_level
            bias_cat_1_name, bias_cat_2_name = bias_categories[bias_cat]  # there should only be 2 since tests are paired
            # validation for if categories are missing
            if not validate_bias_types((bias_cat_1_name,bias_cat_2_name), all_results[steps]):
                continue

            # overall bias
            for steps in all_results.keys():
                bias_cat_1_mean, bias_cat_2_mean, performance_gap, performance_values, p = get_bias_stats(
                    bias_cat_1_name, bias_cat_2_name, all_results[steps])

                # first aggregate
                csv_writer.writerow([bias_cat_1_mean, bias_cat_2_mean, performance_gap, p, bias_cat_1_name,
                                     bias_cat_2_name, bias_cat, steps, test_lang])

                # here we write out just specific label info rather than aggregate
                results_1, results_2 = all_results[steps][bias_cat_1_name], all_results[steps][bias_cat_2_name]
                for idx, vals in enumerate(list(zip(results_1, results_2))):
                    csv_writer_all.writerow([vals[0], vals[1], performance_values[idx], bias_cat_1_name,
                                                 bias_cat_2_name, bias_cat, steps, test_lang])
                for emo in all_results_by_emotion:
                    bias_cat_1_mean, bias_cat_2_mean, performance_gap, performance_values, p = get_bias_stats(
                        bias_cat_1_name, bias_cat_2_name, all_results_by_emotion[emo][steps])

                    results_1, results_2 = all_results_by_emotion[emo][steps][bias_cat_1_name], all_results_by_emotion[emo][steps][bias_cat_2_name]
                    for idx, vals in enumerate(list(zip(results_1, results_2))):
                        csv_writer_all_emo.writerow([vals[0], vals[1], performance_values[idx], bias_cat_1_name,
                                                 bias_cat_2_name, bias_cat, steps, test_lang, emo])

                    csv_emo_writer.writerow([bias_cat_1_mean, bias_cat_2_mean, performance_gap, p, bias_cat_1_name,
                                             bias_cat_2_name, bias_cat, steps, test_lang, emo])








