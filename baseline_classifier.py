import argparse
import os
from pathlib import Path
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import ttest_rel
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn import metrics

import joblib

from datasets import load_dataset

from utils.data_utils import get_label_from_emotion, convert_to_polarity, load_corpus
from evaluation.create_eval_set import lang2bias
from fine_tune_models import scrub




def setup_argparse():
    p = argparse.ArgumentParser()
    p.add_argument('-o', dest='output_dir', default='models/')
    p.add_argument('-l', '--lang', default='zh')
    p.add_argument('-v', '--vector_type', default='bow')
    p.add_argument('--no_save', action='store_true')
    p.add_argument('--scrub', action='store_true')
    p.add_argument('--dummy', choices=["most_frequent", "uniform", "stratified"])
    p.add_argument('--dummy_seed', default=1, type=int, help="dummy seed for use in naming convention")
    return p.parse_args()


def format_data(text):
    full_text = text["review_body"] + text["review_title"]
    label = int(text["stars"])
    return full_text, label


def load_saved(model_dir):
    model_path = os.path.join(model_dir, "model.joblib")
    return joblib.load(model_path)


def eval_baseline_on_corpus(classifier_pipeline, test_corpus):
    emo2bias2results = defaultdict(lambda: defaultdict(list))
    results_performance = defaultdict(lambda: defaultdict(dict))
    for emotion in test_corpus:
        gold_label = get_label_from_emotion(emotion)
        for bias_type in test_corpus[emotion]:
            test_texts = test_corpus[emotion][bias_type]
            predictions = list(classifier_pipeline.predict(test_texts))  # need to cast as list to match BERT output
            emo2bias2results[emotion][bias_type] = predictions

            # convert results into polarity labels: positive (4,5), negative (1-2), neutral (3) (this is what marc paper did but they drop neutral to make it binary)
            polarity_results = list(map(convert_to_polarity, predictions))
            gold_labels = [gold_label] * len(polarity_results)
            results_performance[emotion][bias_type] = {
                "predicted": polarity_results,
                "gold_labels": gold_labels
            }
    bias2results = defaultdict(list)
    bias2performance = defaultdict(lambda: defaultdict(list))
    for emo in emo2bias2results:
        for bias_type in emo2bias2results[emo]:
            bias2results[bias_type].extend(emo2bias2results[emo][bias_type])

            for label_type in ["predicted", "gold_labels"]:
                bias2performance[bias_type][label_type].extend(results_performance[emo][bias_type][label_type])

    return bias2results, emo2bias2results, bias2performance, results_performance

if __name__ == "__main__":
    args = setup_argparse()

    raw_dataset = load_dataset('amazon_reviews_multi', args.lang)

    if args.scrub:
        print("Scrubbing data", file=sys.stderr)
        raw_dataset = raw_dataset.map(scrub)

    split2data = {}
    for split in ["train", "validation"]:
        split2data[split] = [format_data(i) for i in raw_dataset[split]]

    # Run dataset through SGD classifier
    analyzer = "char" if args.lang in ["zh","ja"] else "word"  # TODO could just add a proper tokenizer
    if args.vector_type == "tf-idf":
        vectorizer = TfidfVectorizer(analyzer=analyzer)
    elif args.vector_type == "bow":
        vectorizer = CountVectorizer(analyzer=analyzer)
    else:
        sys.exit("unsupported vector type {}".format(args.vector_type))
    # Classifier params, including defaults
    classifier = SGDClassifier(
        loss="hinge",
        penalty="l2",
        learning_rate="optimal",
        early_stopping=True,
        max_iter=1000,
    )
    if args.dummy:
        print("Using dummy classifier {}".format(args.dummy))
        classifier = DummyClassifier(strategy=args.dummy)

    classifier_pipeline = Pipeline([("vectors", vectorizer), ("classifier", classifier)])
    # train
    classifier_pipeline.fit(*list(zip(*split2data["train"])))

    if not args.no_save:
        # make directory and save model
        output_dir = os.path.join(args.output_dir, f"{args.lang}_0_{args.dummy_seed}/checkpoint/")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = os.path.join(output_dir, "model.joblib")

        joblib.dump(classifier_pipeline, output_path)

        classifier_pipeline = joblib.load(output_path)

    # test - in domain
    print("Validation")
    test_texts, test_labels = list(zip(*split2data["validation"]))

    predictions = classifier_pipeline.predict(test_texts)

    print(metrics.classification_report(test_labels, predictions, labels=range(1,6)))
    print(metrics.confusion_matrix(test_labels, predictions, labels=range(1, 6)))

    # test out of domain
    print("EEC")
    test_corpus = load_corpus(args.lang)

    bias2results, emo2bias2results, bias2performance, results_performance = eval_baseline_on_corpus(classifier_pipeline,
                                                                                                    test_corpus)
    all_labels = ["positive", "negative", "neutral"]
    print("Performance")
    for bias in bias2performance.keys():
        if "rank" in bias:
            continue
        print(bias)
        predictions, test_labels = bias2performance[bias]['predicted'], bias2performance[bias]['gold_labels']
        print(metrics.classification_report(test_labels, predictions, labels=all_labels))
        print(metrics.confusion_matrix(test_labels, predictions, labels=all_labels))
        print("-"*89)

    print("Bias")
    bias_categories = lang2bias[args.lang]
    for bias_cat in bias_categories:
        if "rank" in bias_cat:
            continue
        print(f"\n{bias_cat}\n")
        bias_subtypes = bias_categories[bias_cat]  # there should only be 2 since tests are paired
        print("T-test:")
        statistic, p = ttest_rel(bias2results[bias_subtypes[0]], bias2results[bias_subtypes[1]])
        print("Statistic: {} p: {}".format(statistic, p))
        for subtype in bias_subtypes:
            print(subtype)
            print("Overall Mean: {}".format(np.mean(bias2results[subtype])))








