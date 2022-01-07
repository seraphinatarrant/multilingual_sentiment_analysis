import argparse

import numpy
import spacy
from corpus import defaultdict


seed_words_en = ["angry", "sad", "scared", "happy"]

### JAPANESE
category_words_ja = ["怒り","悲しみ","恐れ","喜び"]
seed_words_ja = ["怒っている", "悲しい","怖がった","嬉しい"]

### SPANISH
# States
seed_words_es_state = ["enojado", "triste", "asustada", "feliz"]
seed_words_es_exp_scared = ["asustado"]

# Situations
seed_words_es_situation = ["molesto", "deprimente", "espantoso", "maravillosa"]
seed_words_es_exp_happy = ["maravilloso"]

###GERMAN
# states
seed_words_de = ["verärgert", "traurig", "erschrocken", "glücklich"]
seed_words_de_exp_angry = ["wütend", "zornig"]
seed_words_de_exp_sad = ["unglücklich", "trist", "unzufrieden"]
seed_words_de_exp_scared = ["ängstlich", "bange", "besorgt"]
seed_words_de_exp_happy = ["zufrieden", "froh", "freudig", "fröhlich", "vergnügt", "heiter"]

all_exp = [seed_words_de_exp_angry, seed_words_de_exp_sad, seed_words_de_exp_scared, seed_words_de_exp_happy]


# situations
seed_words_de = ["irritierend", "deprimierend", "schrecklich", "wundervoll"]
seed_words_de_exp_angry = ["ärgerlich", "lästig", "belästigend"]
seed_words_de_exp_sad = ["bedrückend", "trübselig"]
seed_words_de_exp_scared = ["fürchterlich"]
seed_words_de_exp_happy = ["wunderbar", "herrlich", "wunderschön", "großartig"]
all_exp = [seed_words_de_exp_angry, seed_words_de_exp_sad, seed_words_de_exp_scared, seed_words_de_exp_happy]


lang2model = {
    "de": "de_core_news_lg",
    "ja": "ja_core_news_lg",
    "es": "es_core_news_lg"
}

def setup_argparse():
    p = argparse.ArgumentParser()
    p.add_argument('-l', dest='lang', default='', help='')
    return p.parse_args()


if __name__ == "__main__":
    args = setup_argparse()

    nlp = spacy.load(lang2model[args.lang])
    other_dict = defaultdict(list)

    ## Set up dicts
    # first word will be canonical
    # for en, other, exp in zip(seed_words_en, seed_words_de, all_exp):
    #     other_dict[en].append(other)
    #     other_dict[en].extend(exp)
    seed_words_other = seed_words_es_situation

    for en, other in zip(seed_words_en, seed_words_other): # TODO make this controlled ith args
        other_dict[en].append(other)

    other_exp_dict = {key: set() for key in other_dict.keys()}

    for key in other_dict:
        words = other_dict[key]
        for w in words:
            try:
                w_str = nlp.vocab.strings[w]
                #print(nlp.vocab.vectors[nlp.vocab.strings[w]])
                w_vec = nlp.vocab.vectors[w_str]
            except KeyError:
                print(f"{w} has no vector")
                continue
            most_sim_idx = nlp.vocab.vectors.most_similar(numpy.asarray([w_vec]), n=20)
            most_sim_str = set([nlp.vocab.strings[i].lower() for i in most_sim_idx[0][0]])
            other_exp_dict[key] |= most_sim_str

    for key in other_exp_dict:
        print(f"{key}: {len(other_exp_dict[key])}")
        print(other_exp_dict[key])