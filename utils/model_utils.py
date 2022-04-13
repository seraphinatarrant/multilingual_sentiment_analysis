import yaml
import os
import json
import ipdb

from transformers import AutoTokenizer, BertForSequenceClassification, DistilBertTokenizer, \
    DistilBertForSequenceClassification, pipeline, AutoModelForSequenceClassification, AutoConfig

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

lang2tok_cfg = {
    "ja": "/home/s1948359/multilingual_sentiment_analysis/models/tokenizers/ja",
    "en": "/home/s1948359/multilingual_sentiment_analysis/models/tokenizers/en",
    "zh": "/home/s1948359/multilingual_sentiment_analysis/models/tokenizers/zh",
    "multi": "/home/s1948359/multilingual_sentiment_analysis/models/tokenizers/multi",
    "de": "/home/s1948359/multilingual_sentiment_analysis/models/tokenizers/de",
    "es": "/home/s1948359/multilingual_sentiment_analysis/models/tokenizers/es",
    }

def load_model_and_tokenizer(model_location: str, model_type: str = "", lang: str = "",
                             from_path: bool = False, return_pipeline: bool = False):

    slurm = True
    if not from_path:  # in this case there is a yaml config

        model_loc = yaml.load(open(model_location), Loader=yaml.FullLoader)
        this_model = model_loc[model_type][lang]
        tok_path = this_model
        print(this_model)

        
        if slurm == True:
            this_model = os.path.join('/home/s1948359/multilingual_sentiment_analysis/models/pretrained', this_model)
            cfg = AutoConfig.from_pretrained(this_model)
            tokenizer = AutoTokenizer.from_pretrained(this_model, config=cfg) 
             
    else:
        #ipdb.set_trace()
        this_model = model_location  # there is a checkpoint locally to load
        with open(os.path.join(model_location, "config.json"), "r") as fin:
            cfg = json.load(fin)
            tok_path = cfg["_name_or_path"]

        tokenizer = AutoTokenizer.from_pretrained(tok_path) 
        
    model = AutoModelForSequenceClassification.from_pretrained(this_model,
                                                               num_labels=5,
                                                               problem_type="single_label_classification")

    print("Model and Tokenizer: {} {}".format(this_model, tokenizer))
    # there's an inference error when other tokenizers are use because of an extraneous token_type_ids that the tokenizer returns but distilbert does not use.
    if model.base_model_prefix == "distilbert" and type(tokenizer) != DistilBertTokenizer:
        tokenizer.model_input_names = ['input_ids', 'attention_mask']

    if return_pipeline:
        return pipeline("text-classification", tokenizer=tokenizer, model=model)
    else:
        return model, tokenizer
