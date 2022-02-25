"""
A script to process Amazon US reviews, since they are huge and take ages to download,
and in the process to balance the labels by subsampling, and then make a train test split.
Does not tokenize etc as that is model dependent.

Does scrub if flag is set.
"""

import argparse
import os
from collections import defaultdict
import random

import yaml
import sys

from datasets.utils.info_utils import NonMatchingSplitsSizesError
from tqdm import tqdm

from datasets import load_dataset, interleave_datasets, DatasetDict
from utils.data_utils import us_reviews_cat
from fine_tune_models import scrub

def setup_argparse():
    p = argparse.ArgumentParser()
    p.add_argument('--save', dest='save_loc', default='data/us_reviews', help='place to save dataset')
    p.add_argument('--scrub', action='store_true',
                   help='scrub gender info with regex, only works for english')
    return p.parse_args()

if __name__ == "__main__":
    args = setup_argparse()

    all_datasets = []
    for cat in tqdm(us_reviews_cat):
        print(f"Loading {cat}...")
        try:
            us_dataset = load_dataset('amazon_us_reviews', cat)
        except NonMatchingSplitsSizesError:
            try:  # file may be corrupted, try to redownload
                us_dataset = load_dataset('amazon_us_reviews', cat,
                                          download_mode="force_redownload")
            except:  # if this happens frequently it could be a bad config so could try ignore_verifications=True, but then have to check that the dataset is non-empty + valid another way
                print(f"Skipping {cat}")
                continue
        all_datasets.append(us_dataset)

    master_dataset = interleave_datasets([d["train"] for d in all_datasets])

    ### now balance the labels
    print("Balancing...")
    balanced_dataset = []
    # sort them into 5 buckets for each label
    star_buckets = defaultdict(list)
    for entry in tqdm(master_dataset):
        star_buckets[entry["star_rating"]].append(entry)

    # take the entirety of the smallest one
    max_samples = min([len(vals) for vals in star_buckets.values()])
    for key in star_buckets:
        balanced_dataset.extend(random.sample(star_buckets[key], max_samples))

    remaining_per = len(balanced_dataset)/len(master_dataset["train"])*100
    print(f"Final Dataset is {remaining_per}% of original.")

    new_dataset = DatasetDict({"train": balanced_dataset})

    # 5% test + eval
    train_test = new_dataset["train"].train_test_split(test_size=0.05, seed=42)
    # split again
    test_valid = train_test['test'].train_test_split(test_size=0.5, seed=42)
    # gather into DatasetDict
    new_dataset = DatasetDict({
        'train': train_test['train'],
        'test': test_valid['test'],
        'validation': test_valid['train']})

    if args.scrub:
        print("Scrubbing data", file=sys.stderr)
        new_dataset = new_dataset.map(scrub)

    print("Saving...")
    if not os.path.exists(args.save_loc):
        os.makedirs(args.save_loc)
    new_dataset.save_to_disk(args.save_loc)