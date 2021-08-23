"""
Reads in a csv of information and creates an object (dataframe? transformers corpus?) with expanded sentences.

Language specific things:
Japanese has different adjectival forms for emotions, but not situation/state distinction not case marking.
German has case marking and gender, but the only emotion distinctions are situation/state.


"""
import argparse
import re
from collections import defaultdict
from typing import List, Dict

import pandas as pd

SHEET_NAMES = ["Overview Data ({})", "People ({})", "Emotions ({})"]
EMOTION_HEADERS = ["Emotion Category", "Emotion Word", "Grammar (situation, state)"]
OVERVIEW_HEADERS = ["Template Sentences"]
PEOPLE_HEADERS = ["Person 1", "Person 2", "Person 1 Type", "Person 2 Type", "Bias type", "Grammar (Subject, Object)"]

EMOTION_CATEGORIES = ["anger", "joy", "sadness", "fear"]

iso2name = {
    "ja": "JPN",
    "de": "German"
}
lang2bias = {

}

zero_out = ["non-marked"]

grammar2pattern = {
    "situation": "<emotional situation word>",
    "state": "<emotion word>",
    "Subject": "<person subject>",
    "Object": "<person object>"
}


def setup_argparse():
    p = argparse.ArgumentParser()
    p.add_argument('-l', dest='lang', default='ja', help='')
    p.add_argument('-x', dest='xls_file', default='evaluation/Equity-Evaluation-Corpus.xlsx')
    return p.parse_args()


def filter_by_pattern(all_sents, pattern: str) -> List:
    re_pattern = re.compile(pattern)
    filtered_sents = [sent for sent in all_sents if re_pattern.search(sent)]
    #print("Found {} sents for pattern {}".format(len(filtered_sents), pattern))
    return filtered_sents


def fill_emotions(template_sents, emotion_sheet) -> Dict[str, set]:
    cat2sents = defaultdict(set)
    # find all emotion patterns, and replace with appropriate emotion words

    # filter sentences for appropriate pattern
    for grammar_type in ["situation", "state"]:
        sents = filter_by_pattern(template_sents, grammar2pattern[grammar_type])
        # filter by appropriate emotion grammar
        mask = emotion_sheet[EMOTION_HEADERS[2]].str.contains(grammar_type)  # uses the regex for the grammar form
        filtered_emotion_sheet = emotion_sheet[mask]

        for cat in EMOTION_CATEGORIES:
            mask = filtered_emotion_sheet[EMOTION_HEADERS[0]].str.contains(cat)  # filter by emotion category
            this_emotion_cat = filtered_emotion_sheet[mask]
            emotion_words = this_emotion_cat.get(EMOTION_HEADERS[1])  # should be just the words column
            for sent in sents:
                for word in emotion_words:
                    new_sent = re.sub(grammar2pattern[grammar_type], word, sent)
                    cat2sents[cat].add(new_sent)
    return cat2sents


def fill_people(cat2sents: Dict, people_sheet) -> Dict[str, Dict[str,List]]:
    bias2cat2sents = defaultdict(lambda: defaultdict(list))
    pairs = set() # I could instead structurally store them as tuple pairs rather than coindexed lists?
    for grammar_type in ["Subject", "Object"]:
        # filter by people grammar
        mask = people_sheet[PEOPLE_HEADERS[5]].str.contains(grammar_type)
        filtered_people_sheet = people_sheet[mask]
        for emotion_cat in cat2sents:
            sents = filter_by_pattern(cat2sents[emotion_cat], grammar2pattern[grammar_type])
            for sent in sents:
                for row in filtered_people_sheet.values:
                    row = [x if x != "non-marked" else "" for x in row]  # TODO remember to strip whitespace in this case
                    person1, person2, bias_cat1, biascat_2, bias_type = row[:5]

                    new_sent1 = re.sub(grammar2pattern[grammar_type], person1, sent)
                    new_sent2 = re.sub(grammar2pattern[grammar_type], person1, sent)
                    if (new_sent1, new_sent2) in pairs:
                        continue
                    #  Need to handle duplication when something is both grammar categories
                    #  since in some pairs the counterfactual will be duplicated for one category
                    #  TODO work out if this duplication is a problem
                    pairs.add((new_sent1, new_sent2))
                    bias2cat2sents[bias_cat1][emotion_cat].append(new_sent1)
                    bias2cat2sents[biascat_2][emotion_cat].append(new_sent2)

    return bias2cat2sents

if __name__ == "__main__":
    args = setup_argparse()

    all_sheets = pd.read_excel(args.xls_file, sheet_name=None)

    #print(SHEET_NAMES[0].format(iso2name[args.lang]))  # .get(OVERVIEW_HEADERS[0])
    #print(all_sheets.keys())
    template_sents = all_sheets.get(SHEET_NAMES[0].format(iso2name[args.lang])).get(OVERVIEW_HEADERS[0])
    # First expand sentence with emotions
    emotion_sheet = all_sheets.get(SHEET_NAMES[2].format(iso2name[args.lang]))
    test_sents = fill_emotions(template_sents, emotion_sheet)  # dict of emotion category to set of sentences
    # then for each bias type, expand with people and keep in contrastive pairs
    people_sheet = all_sheets.get(SHEET_NAMES[1].format(iso2name[args.lang]))
    test_sents = fill_people(test_sents, people_sheet)  # nested dict of bias cat to emotion cat to list of sents: must be a list because it is paired

    for key in test_sents:
        print(f"{key}:")
        for key2 in test_sents[key]:
            #print(f"{key2}: {len(test_sents[key][key2])}")
            print(f"{key2}:")
            for sent in test_sents[key][key2]:
                print(sent)





