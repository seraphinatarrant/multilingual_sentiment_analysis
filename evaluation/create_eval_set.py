"""
Reads in a csv of information and creates an object (dataframe? transformers corpus?) with expanded sentences.

HOW TO USE:
The final corpus will be emotion: bias type: list of sentences, where the lists are ordered to be minimal pairs
The preamble to the script is a primitive config (should be a real config) where there are dictionaries with patterns
that are important, so you have to set iso2name, lang2bias, lang2grammar (for the grammar restrictions) and grammar2pattern for the actual regexes

Language specific things:
Japanese has different adjectival forms for emotions, but not situation/state distinction not case marking.
German has case marking and gender, but the only emotion distinctions are situation/state.


"""
import argparse
import sys
import json
import re
from collections import defaultdict
from typing import List, Dict

import pandas as pd

SHEET_NAMES = ["Overview Data ({})", "People ({})", "Emotions ({})"]
EMOTION_HEADERS = ["Emotion Category", "Emotion Word", "Grammar"]
OVERVIEW_HEADERS = ["Template Sentences"]
PEOPLE_HEADERS = ["Person 1", "Person 2", "Person 1 Type", "Person 2 Type", "Bias type", "Grammar"] # TODO add validation on xlsx that this is the order
# Spanish Grammar 1 (m/f)	Grammar 2 (m/f) for people and Grammar (f/m) for emotions

EMOTION_CATEGORIES = ["anger", "joy", "sadness", "fear"]
NO_EMOTION_CATEGORY = "no_emotion"
default_person_placeholder = "<person>"

iso2name = {
    "ja": "JPN",
    "de": "German",
    "es": "Spanish",
    "zh": "Chinese",
    "en": "English"
}

# just used for validation -- the privileged must come before minoritised
lang2bias = {
    "ja": {
        "gender": ["male", "female"],
        "rank": ["rank: privileged", "rank: minoritized"],
        "race": ["race: privileged", "race: minoritized"]
    },
    "de": {
        "gender": ["male", "female"],
        "race": ["race: privileged", "race: minoritized"]
    },
    "es": {
        "gender": ["male", "female"]
    },
    "zh": {
        "gender": ["male", "female"]
    },
    "en": {
        "gender": ["male", "female"],
        "race": ["race: privileged", "race: minoritized"]
    }

}

zero_out = "non-marked"

lang2grammar = {
    "ja": {
            "people": ["subject", "object"],
            "emotions": ["active", "passive"]
    },
    "de": {
        "people": ["subject", "acc. object", "dat. object"],
        "emotions": ["situation", "state"]
    },
    "es": {
        "people": ["subject", "object"],
        "emotions": ["state_estoy", "state_tengo", "situation_female", "situation_male", "situation_male_plural"]
    },
    "zh": {
            "people": ["subject", "object"],
            "emotions": ["situation", "state"]
    },
    "en": {
            "people": ["subject", "object"],
            "emotions": ["situation_a", "situation_an", "state"]
    },
    "Other": {
        "people": [],
        "emotions": []
    }
}

grammar2pattern = {
    "situation": "<emotional situation word>",
    "state": "<emotion word>",
    "subject": "<person subject>",
    "object": "<person object>",
    "acc. object": "<person acc. object>",
    "dat. object":  "<person dat. object>",
    "active": "<emotion word active>",
    "passive": "<emotion word passive>",
    "state_estoy": "<emotion word A>",
    "state_tengo": "<emotion word B>",
    "situation_male_plural": "<emotional situation word plural>",
    "situation_female": "<emotional situation word female>",
    "situation_male": "<emotional situation word male>",
    "situation_a": "<emotional situation word a>",
    "situation_an": "<emotional situation word an>",
    "male": "<person male>",
    "female": "<person female>"
}

class Sentence:
    def __init__(self, sent_string):
        self.text = sent_string
        self.grammar2string = {} # male: string, female: string, etc

    def __call__(self):
        return self.text

    def __repr__(self):
        return self.text

    def __str__(self):
        return self.text


def setup_argparse():
    p = argparse.ArgumentParser()
    p.add_argument('-l', dest='lang', choices=["ja", "de", "es", "zh", "en"])
    p.add_argument('-x', dest='xls_file', default='evaluation/Equity-Evaluation-Corpus.xlsx')
    p.add_argument('-o', dest='output', default="evaluation/corpora/EEC_{}.json", help='filename of test corpus output')
    p.add_argument('--print', action='store_true')
    return p.parse_args()


def filter_by_pattern(all_sents, pattern: str) -> List:
    re_pattern = re.compile(pattern)
    if type(list(all_sents)[0]) == str:
        filtered_sents = [sent for sent in all_sents if re_pattern.search(sent)]
    else: #assume is Sentence object
        filtered_sents = [sent for sent in all_sents if re_pattern.search(sent.text)]
    #print("Found {} sents for pattern {}".format(len(filtered_sents), pattern))
    return filtered_sents

# def replace_person_tag(sentence: str) -> set[str]:
#     new_sentences = set()
#     for person_type in lang2grammar[lang]["people"]:
#         ns = re.sub(default_person_placeholder, grammar2pattern[person_type], sentence)
#         new_sentences.add(ns)
#     return new_sentences

def fill_emotions(template_sents, emotion_sheet, lang, person_emotion_agreement=False) -> Dict[str, set]:
    cat2sents = defaultdict(set)
    # find all emotion patterns, and replace with appropriate emotion words

    used_sentences = set()
    # filter sentences for appropriate pattern
    for grammar_type in lang2grammar[lang]["emotions"]:
        # currently person_emotion_agreement is spanish
        # For es, if type is situation, will agree with the fixed subject, if the type is state, will agree with changeable person tag
        # so in the former case, pick *one* form that agrees, in the latter create multiple variants for each possible person gender and change the default tag

        sents = filter_by_pattern(template_sents, grammar2pattern[grammar_type]) # filters for everything with the appropriate patterns
        used_sentences |= set(sents) # for including unused ones later
        # filter by appropriate emotion grammar
        mask = emotion_sheet["Grammar"].str.contains(grammar_type, na=False)  # uses the regex for the grammar form in the Grammar column so only have words that match
        filtered_emotion_sheet = emotion_sheet[mask]

        for cat in EMOTION_CATEGORIES:
            mask = filtered_emotion_sheet["Emotion Category"].str.contains(cat, na=False)  # filter by emotion category (anger, sadness, etc)
            this_emotion_cat = filtered_emotion_sheet[mask]
            #for row in this_emotion_cat.iterrows():
            #    print(row)
            #emotion_words = this_emotion_cat.get("Emotion Word")  # should be just the words column
            #if person_emotion_agreement:
            #    emotion_gender = this_emotion_cat.get(EMOTION_ES) # Get the gender of the emotion
            for sent in sents:
                for row in this_emotion_cat.iterrows():
                    word = row[1].get("Emotion Word")
                    new_sent = Sentence(re.sub(grammar2pattern[grammar_type], word, sent))
                    if person_emotion_agreement and grammar_type == "state_estoy": # tengo doesn't need the difference
                        alternate_word = row[1].get("Emotion Word female")
                        new_sent.grammar2string["male"] = new_sent.text
                        new_sent.grammar2string["female"] = re.sub(grammar2pattern[grammar_type], alternate_word, sent)
                    cat2sents[cat].add(new_sent)
    # Need to add a capture all the remainder sentences that have no emotion categories
    leftover_sents = set(template_sents) - used_sentences
    cat2sents[NO_EMOTION_CATEGORY] = set(Sentence(s) for s in leftover_sents)

    return cat2sents


def fill_people(cat2sents: Dict, people_sheet, lang:str, person_emotion_agreement=False) -> Dict[str, Dict[str,List]]:
    cat2bias2sents = defaultdict(lambda: defaultdict(list))
    pairs = set() # I could instead structurally store them as tuple pairs rather than coindexed lists?

    for grammar_type in lang2grammar[lang]["people"]:
        # filter by people grammar
        mask = people_sheet["Grammar"].str.contains(grammar_type, na=False)
        filtered_people_sheet = people_sheet[mask]
        for emotion_cat in cat2sents:
            sents = filter_by_pattern(cat2sents[emotion_cat], grammar2pattern[grammar_type])
            for sent in sents:
                for row in filtered_people_sheet.values:
                    row = [x if x != zero_out else "" for x in row] # deals with unmarked phenomenon
                    person1, person2, bias_cat1, bias_cat2, bias_type = map(str.strip,row[:5])
                    # make sure neither empty
                    if not person1 or not person2:
                        print("one of the pairs in the following row is empty:\n {}".format(row[:5]))
                        continue

                    if person_emotion_agreement and bool(sent.grammar2string): # match the person1 and two types to their grammar. Sometimes sentences don't have two variations even with agreement.
                        person1_type, person2_type = map(str.strip, row[6:8])
                        # the grammar type values should be keys in the Sentence grammar2string dict
                        new_sent1 = re.sub(grammar2pattern[grammar_type], person1, sent.grammar2string[person1_type])
                        new_sent2 = re.sub(grammar2pattern[grammar_type], person1, sent.grammar2string[person2_type])
                    else:
                        new_sent1 = re.sub(grammar2pattern[grammar_type], person1, sent.text)
                        new_sent2 = re.sub(grammar2pattern[grammar_type], person2, sent.text)

                    # deal with whitespace issues
                    new_sent1 = re.sub("\s+", " ", new_sent1)
                    new_sent2 = re.sub("\s+", " ", new_sent2)
                    if (new_sent1, new_sent2) in pairs:
                        continue
                    #  Need to handle duplication when something is both grammar categories
                    #  since in some pairs the counterfactual will be duplicated for one category
                    #  TODO work out if this duplication is a problem
                    pairs.add((new_sent1, new_sent2))
                    cat2bias2sents[emotion_cat][bias_cat1].append(new_sent1)
                    cat2bias2sents[emotion_cat][bias_cat2].append(new_sent2)

    return cat2bias2sents

def validate_pairs(corpus, lang):
    print("Validating Corpus Samples...", file=sys.stderr)
    bias_categories = lang2bias[lang]
    for bcat in bias_categories:
        print(f"For bias category: {bcat}", file=sys.stderr)
        sub_types = bias_categories[bcat]
        for emo in corpus:
            len0, len1 = len(corpus[emo][sub_types[0]]), len(corpus[emo][sub_types[1]])
            assert len0 == len1, f"Length of samples does not match for {emo} for {sub_types}: " \
                                 f"{sub_types[0]}: {len0}, {sub_types[1]}: {len1}"
    print("Done.", file=sys.stderr)



if __name__ == "__main__":
    args = setup_argparse()

    all_sheets = pd.read_excel(args.xls_file, sheet_name=None)

    # setup
    template_sheet = all_sheets.get(SHEET_NAMES[0].format(iso2name[args.lang]))
    person_emotion_agreement = True if args.lang in ["es"] else False
    emotion_sheet = all_sheets.get(SHEET_NAMES[2].format(iso2name[args.lang]))
    people_sheet_name = SHEET_NAMES[1].format(iso2name[args.lang])
    people_sheet = all_sheets.get(people_sheet_name)

    # Validate that columns for people are in the correct order (only the first 6 matter)
    people_headers = list(people_sheet.columns)
    assert people_headers[:6] == PEOPLE_HEADERS, f"Columns in {people_sheet_name} do " \
                                             f"not match correct order:\n" \
                                             f"Correct: {PEOPLE_HEADERS}\n Actual: {people_headers}"

    #print(SHEET_NAMES[0].format(iso2name[args.lang]))  # .get(OVERVIEW_HEADERS[0])
    #print(all_sheets.keys())
    template_sents = template_sheet.get(OVERVIEW_HEADERS[0])
    # First expand sentence with emotions
    test_sents = fill_emotions(template_sents, emotion_sheet, args.lang, person_emotion_agreement)  # dict of emotion category to set of sentences

    # then for each bias type, expand with people and keep in contrastive pairs
    people_sheet = all_sheets.get(SHEET_NAMES[1].format(iso2name[args.lang]))

    test_sents = fill_people(test_sents, people_sheet, args.lang, person_emotion_agreement)  # nested dict of emotion cat to bias cat to list of sents: must be a list because it is paired

    validate_pairs(test_sents, args.lang)

    if args.print:
        for key in test_sents:
            print(f"{key}:")
            for key2 in test_sents[key]:
                #print(f"{key2}: {len(test_sents[key][key2])}")
                print(f"{key2}:")
                for sent in test_sents[key][key2]:
                    print(sent)

    with open(args.output.format(args.lang), "w") as fout:  # Note that a bunch of these characters will be unicode
        json.dump(test_sents, fout)







