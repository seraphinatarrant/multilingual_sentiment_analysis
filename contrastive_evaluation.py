import argparse
import json
from collections import defaultdict
from typing import List

from scipy.stats import ttest_rel
import numpy as np
from transformers import AutoTokenizer, BertForSequenceClassification, pipeline
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from utils.model_utils import load_model_and_tokenizer
from utils.data_utils import get_label_from_emotion, convert_to_polarity
from evaluation.create_eval_set import lang2bias


def setup_argparse():
    p = argparse.ArgumentParser()
    p.add_argument('-l', dest='lang', choices=['en', 'de', 'ja', 'es', "multi", "zh"], help='language of test set')
    p.add_argument('--target_lang', help='target lang to use if different from source lang')
    p.add_argument('--translated_from', choices=['de', 'ja', 'es'],
                   help='if test set is translated from another language, which one')
    p.add_argument('--model_loc', default="config/model_loc.yaml", help="yaml of locations of all the models")
    return p.parse_args()


def eval_model_on_corpus(classifier_pipeline, test_corpus, return_performance=False):
    """test_corpus is a json text corpus saved with create_eval_set
    classifier_pipeline is a huggingface pipeline with model and tokenizer
    returns a dict of {bias_type: all_scores} and a second dict of {emotion: bias_type: all_results}"""
    # evaluate model and calculate metric
    emo2bias2results = defaultdict(lambda: defaultdict(list))
    results_performance = defaultdict(lambda: defaultdict(dict))
    label2id = classifier_pipeline.model.config.label2id

    def postprocess(results: dict) -> int:
        return label2id.get(results['label']) + 1  # The +1 adjustment undoes the 0 indexing TODO could have done with ClassLabel

    # For each type of bias
    # metrics:
    # overall average predicted score for one bias type vs another bias type
    # also broken out by emotion type
    # pairwise
    # can use a paired two sample t-test for significance -  original applied Bonferroni correction since tested lots of systems, should I apply it since I am using N systems (where N is type of bias)?
    for emotion in test_corpus:
        gold_label = get_label_from_emotion(emotion)
        for bias_type in test_corpus[emotion]:
            #TODO pick up here working out how to get output
            sentences = test_corpus[emotion][bias_type]
            all_results = list(map(postprocess, classifier_pipeline(sentences)))
            emo2bias2results[emotion][bias_type] = all_results

            # convert results into polarity labels: positive (4,5), negative (1-2), neutral (3) (this is what marc paper did but they drop neutral to make it binary)
            polarity_results = list(map(convert_to_polarity, all_results))
            gold_labels = [gold_label] * len(polarity_results)
            results_performance[emotion][bias_type] = {
                "predicted": polarity_results,
                "gold_labels": gold_labels
            }

    # print stuff: overall mean, by emotion mean, statistical test
    bias2results = defaultdict(list)
    bias2performance = defaultdict(lambda: defaultdict(list))
    for emo in emo2bias2results:
        for bias_type in emo2bias2results[emo]:
            bias2results[bias_type].extend(emo2bias2results[emo][bias_type])

            for label_type in ["predicted", "gold_labels"]:
                bias2performance[bias_type][label_type].extend(results_performance[emo][bias_type][label_type])

    if return_performance:
        return bias2results, emo2bias2results, bias2performance, results_performance
    else:
        return bias2results, emo2bias2results


if __name__ == "__main__":
    args = setup_argparse()

    # load model

    model, tokenizer, model_path = load_model_and_tokenizer("fine_tuned_models", args.lang, args.model_loc,
                                                            local_model=True)
    print(f"Loading model {model_path}")
    classifier_pipe = pipeline("text-classification", tokenizer=tokenizer, model=model)

    data_lang = args.target_lang if args.lang == "multi" else args.lang
    # load eval corpus, currently in json TODO make it a huggingface Dataset
    eval_corpus_path = "evaluation/corpora/EEC_{}.json"
    with open(eval_corpus_path.format(data_lang), "r") as fin:
        test_corpus = json.load(fin)

    # prepare corpus according to model
    bias_categories = lang2bias[data_lang]
    bias2results, results, bias2performance, results_performance = eval_model_on_corpus(classifier_pipe, test_corpus)

    # print and write emotion and  for overall
    all_gold, all_pred = [], []
    for bias_cat in bias_categories:
        print(f"\n{bias_cat}\n")
        bias_subtypes = bias_categories[bias_cat]  # there should only be 2 since tests are paired
        print("T-test:")
        statistic, p = ttest_rel(bias2results[bias_subtypes[0]], bias2results[bias_subtypes[1]])
        print("Statistic: {} p: {}".format(statistic, p))

        print("Overall Performance")
        gold, pred = [], []
        for subtype in bias_subtypes:
            gold.extend(bias2performance[subtype]["gold_labels"])
            pred.extend(bias2performance[subtype]["predicted"])

        all_gold.extend(gold)
        all_pred.extend(pred)

        precision, recall, f1, support = precision_recall_fscore_support(gold, pred, average="macro")
        acc = accuracy_score(gold, pred)
        print(f"Precision: {precision}\n Recall: {recall}\n F1: {f1}\n Acc: {acc}\n")

        for subtype in bias_subtypes:
            print(subtype)
            print("Overall Mean: {}".format(np.mean(bias2results[subtype])))
            print("By Emotion Mean:")
            for emotion in results:
                print("{}: {}".format(emotion, np.mean(results[emotion][subtype])))

            print("\nPerformance\n")
            gold, pred = bias2performance[subtype]["gold_labels"], bias2performance[subtype]["predicted"]
            precision, recall, f1, support = precision_recall_fscore_support(gold, pred, average="macro")
            acc = accuracy_score(gold, pred)
            print(f"\n{subtype}\n")
            print(f"Precision: {precision}\n Recall: {recall}\n F1: {f1}\n Acc: {acc}\n")
        print("-" * 89)

    print("\nALL SCORES\n")
    precision, recall, f1, support = precision_recall_fscore_support(all_gold, all_pred, average="macro")
    acc = accuracy_score(all_gold, all_pred)
    print(f"Precision: {precision}\n Recall: {recall}\n F1: {f1}\n Acc: {acc}\n")

#     ### Calc performance here # TODO also calc precision and recall difference between bias types
#     print("\nPerformance\n")
#     for bias_cat in bias_categories:
#         print(f"\n{bias_cat}\n")
#         bias_subtypes = bias_categories[bias_cat]  # there should only be 2 since tests are paired
#         for bias_subtype in bias_subtypes:
#             gold, pred = bias2performance[bias_subtype]["gold_labels"], bias2performance[bias_subtype]["predicted"]
#             precision, recall, f1, support = precision_recall_fscore_support(gold, pred, average="macro")
#             acc = accuracy_score(gold, pred)
#             print(f"\n{bias_subtype}\n")
#             print(f"Precision: {precision}\n Recall: {recall}\n F1: {f1}\n Acc: {acc}\n")


# TODO find particular errors that are obvious for analysis where the valence is wrong.
# TODO first get the performance by emotion stuff
# then go through the by emotion results, and each should have a clear neg/pos/neutral results, and then find ones that are way off
# so for get_label_from_emotion -> positive look at 1,2, for negative look at 4,5, and for neutral look at 1,5.











