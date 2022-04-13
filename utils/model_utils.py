import yaml
import os
import json
from transformers import AutoTokenizer, BertForSequenceClassification, DistilBertTokenizer, \
    DistilBertForSequenceClassification, pipeline, AutoModelForSequenceClassification

# Stores number of steps at which each model converged

lang2convergence = {
    "de": 35720, #36720,
    "es": 35720, #35190, #OR 41310, 0.61
    "ja": 42864, #44370,
    "zh": 35720, #35190,
    "en": 28576, #27050,
    "en_scrubbed": 25000, #26520
}
multi_convergence = 100644
balanced_multi_convergence = 75000

compressed2convergence = {
    "de": 52621,
    #"es": ,
    "ja": 60436,
    "zh": 43750,
    "en": 44285,
    "en_scrubbed": 17193
}
compressed_multi_convergence = 78280
compressed_balanced_multi_convergence = 33336

def load_model_and_tokenizer(model_location: str, model_type: str = "", lang: str = "",
                             from_path: bool = False, return_pipeline: bool = False):
    if not from_path:  # in this case there is a yaml config

        model_loc = yaml.load(open(model_location), Loader=yaml.FullLoader)
        this_model = model_loc[model_type][lang]
        tok_path = this_model
        print(this_model)

    else:
        this_model = model_location  # there is a checkpoint locally to load
        with open(os.path.join(model_location, "config.json"), "r") as fin:
            cfg = json.load(fin)
            tok_path = cfg["_name_or_path"]

    tokenizer = AutoTokenizer.from_pretrained(tok_path)  # this way it autodetects the tokenizer from pretrained model
    model = AutoModelForSequenceClassification.from_pretrained(this_model,
                                                               num_labels=5,
                                                               problem_type="single_label_classification")

    # there's an inference error when other tokenizers are use because of an extraneous token_type_ids that the tokenizer returns but distilbert does not use.
    if model.base_model_prefix == "distilbert" and type(tokenizer) != DistilBertTokenizer:
        tokenizer.model_input_names = ['input_ids', 'attention_mask']

    if return_pipeline:
        return pipeline("text-classification", tokenizer=tokenizer, model=model)
    else:
        return model, tokenizer
