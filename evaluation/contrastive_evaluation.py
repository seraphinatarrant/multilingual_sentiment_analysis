import argparse

from transformers import AutoTokenizer, AutoModelForMaskedLM


def setup_argparse():
    p = argparse.ArgumentParser()
    p.add_argument('-m', dest='model', default='', help='model to use for evaluation')
    p.add_argument('-l', dest='lang', choices=['en', 'de', 'ja', 'es'], help='language of test set')
    p.add_argument('--translated_from', choices=['de', 'ja', 'es'],
                   help='if test set is translated from another language, which one')
    return p.parse_args()


if __name__ == "__main__":
    args = setup_argparse()

    # load model
    tokenizer = AutoTokenizer.from_pretrained()
    model = AutoModelForMaskedLM.from_pretrained("bert-base-german-cased")

    # load eval corpus

    # evaluate model and calculate metric




