import argparse
import yaml

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import torch
from datasets import load_dataset, load_metric, ClassLabel
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

labels = ClassLabel(names=[str(i) for i in range(1, 6)])


def compute_metrics(pred):
    metric = load_metric("accuracy")
    gold_labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # macro average ok since dataset should be balanced, need to change if change datasets
    precision, recall, f1, support = precision_recall_fscore_support(gold_labels, preds, average='macro')
    acc = metric.compute(references=gold_labels, predictions=preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def tokenize_function(examples):
    tokens = tokenizer(examples["review_body"], examples["review_title"], padding="max_length", truncation=True,
                     max_length=500)
    tokens["labels"] = labels.str2int(examples["stars"])
    return tokens

def setup_argparse():
    p = argparse.ArgumentParser()
    p.add_argument('-l', dest='lang', help='')
    p.add_argument('--model_loc', default="models/model_loc.yaml", help="yaml of locations of all the models")
    return p.parse_args()

if __name__ == "__main__":
    args = setup_argparse()
    model_loc = yaml.load(open(args.model_loc), Loader=yaml.FullLoader)
    this_model = model_loc["base_models"][args.lang]

    tokenizer = AutoTokenizer.from_pretrained(this_model)
    model = BertForSequenceClassification.from_pretrained(this_model,
                                                               num_labels=5,
                                                               problem_type="single_label_classification")
    print(model)

    # TODO add support for multilingual fine tuning
    raw_datasets = load_dataset('amazon_reviews_multi', args.lang)
    # Tokenize dataset, for amazon multi the sections are review_body and review_title
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    # Rename fields
    #tokenized_datasets = tokenized_datasets.map(lambda examples: {'labels': examples['stars']}, batched=True)

    # format dataset object types
    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    # .map(ClassLabel

    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
    full_train_dataset = tokenized_datasets["train"]
    full_eval_dataset = tokenized_datasets["test"]
    print(full_train_dataset.column_names)

    # fine tune
    training_args = TrainingArguments(output_dir="models/", evaluation_strategy="epoch", save_strategy="epoch",
                                      logging_strategy="epoch", save_total_limit="5", load_best_model_at_end=True)
    trainer = Trainer(
        model=model, args=training_args, train_dataset=small_train_dataset, eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.evaluate()

    # save pretrained model (and tokenizer?) and make sure that it's


