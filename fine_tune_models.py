import argparse
import re
import sys
from math import ceil

import yaml
import os
from datetime import datetime
from tqdm import tqdm

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import torch
from datasets import load_dataset, load_metric, ClassLabel, interleave_datasets, concatenate_datasets, DatasetDict, \
    load_from_disk
from datasets.utils.info_utils import NonMatchingSplitsSizesError
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, AdamW, \
    get_constant_schedule_with_warmup, get_linear_schedule_with_warmup
import wandb

from utils.model_utils import load_model_and_tokenizer
from utils.data_utils import us_reviews_cat, us_reviews_columns, marc_reviews_colummns
from utils.custom_trainer import CustomTrainer

labels = ClassLabel(names=[str(i) for i in range(1, 6)])

local_metric_path = "/home/s1948359/multilingual_sentiment_analysis/evaluation/accuracy.py"

def compute_metrics(pred, offline=False):
    if offline:
        metric = load_metric(local_metric_path)
    else:
        metric = load_metric("accuracy")
    gold_labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # macro average ok since dataset should be balanced, need to change if change datasets
    precision, recall, f1, support = precision_recall_fscore_support(gold_labels, preds, average='macro')
    acc = metric.compute(references=gold_labels, predictions=preds)
    return {
        'accuracy': acc.get("accuracy"),
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def scrub(text, replacement="_"):
    regExp = r"\b(?:[Hh]e|[Ss]he|[Hh]er|[Hh]is|[Hh]im|[Hh]ers|[Hh]imself|[Hh]erself|[Mm][Rr]|[Mm][Rr][sS]|[Mm][Ss]|[Ww]ife|[Hh]usband|[Dd]augher|[Ss]on|[Ww]oman|[Mm]an|[Gg]irl|[Bb]oy|[Ss]ister|[Bb]rother|[Mm]other|[Ff]ather|[Mm]om|[Dd]ad|[Aa]unt|[Uu]ncle|[Mm]a|[Pp]a|[Gg]irlfriend|[Bb]oyfriend)\b"
    sections = ["review_body", "review_title"] if "review_title" in text else ["review_body", "review_headline"]
    for section in sections:
        s, n = re.subn(regExp, replacement, text[section])
        text[section] = s
        if n > 0:
            print(n, file=sys.stderr)
    return text

def tokenize_function_marc(examples):
    tokens = tokenizer(examples["review_body"], examples["review_title"], padding="max_length", truncation=True,
                       max_length=500)
    tokens["labels"] = labels.str2int(examples["stars"])
    return tokens


def tokenize_function_us(examples):
    tokens = tokenizer(examples["review_body"], examples["review_headline"], padding="max_length", truncation=True,
                       max_length=500)
    tokens["labels"] = labels.str2int(examples["star_rating"])
    return tokens


def setup_argparse():
    p = argparse.ArgumentParser()
    p.add_argument('-l', '--lang', choices=['de', 'ja', 'es', 'multi', 'en', 'zh', 'fr'], dest='lang', help='')
    p.add_argument('--target_lang', help='target lang to use if different from source lang')
    p.add_argument('--model_loc', default="config/model_loc.yaml", help="yaml of locations of all the models")
    p.add_argument('--epochs', type=float, default=3, help="number of epochs to run")
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--mono_lr', type=float, default=8e-7)
    p.add_argument('--transfer_lr', type=float, default=5e-6)
    p.add_argument('--small_test', action='store_true', help='run script on a fraction of the training set')
    p.add_argument('--use_product_cat', action='store_true')  # TODO implement using product category
    p.add_argument('--evaluate_only', action='store_true')
    p.add_argument('--model_output', type=str, default='~/multilingual_sentiment_analysis/{}/{}_{}_{}_{}')
    p.add_argument('--load_model', type=str, help="load an already pretrained model")
    p.add_argument('--dataset_loc', default='/home/ec2-user/SageMaker/efs/sgt/data/')
    p.add_argument('--load_saved_dataset', action='store_true')
    p.add_argument('--project_name', default="fine_tuning_v3", help='name for wandb project')
    p.add_argument('--classifier_dropout', type=float, default=0.1, help='dropout for classifier')
    p.add_argument('--compressed', action='store_true', help='whether to use a compressed model')
    p.add_argument('--offline', action='store_true', help='work without internet access')
    p.add_argument('--scrub', action='store_true', help='scrub gender info with regex, only works for english')
    p.add_argument('--xl_transfer', action='store_true', help='use large amounts of english data for training')
    return p.parse_args()


if __name__ == "__main__":
    args = setup_argparse()
    print("Training with args:")
    print(args)

    model_type = "compressed_models" if args.compressed else "models"
    lang = args.lang if args.lang != "multi" else "{}+{}".format(args.lang, args.target_lang)
    time_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    offline = args.offline
    
    if args.load_model:
        model_path = args.load_model
        model, tokenizer = load_model_and_tokenizer(model_path, from_path=True)
        print(f"Loaded model from: {model_path}")
        # get the higher level dir for saving
        model_dir, _ = os.path.split(model_path)
        model_output = os.path.join(args.model_output.format(model_type, lang, args.epochs, args.seed, time_now), "resumed")
        log_output = os.path.join(model_output, "logs")

    else:
        model_type = "compressed_models" if args.compressed else "models"
        time_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        model_output = args.model_output.format(model_type, lang, args.epochs, args.seed, time_now)
        log_output = os.path.join(model_output, "logs")
        model, tokenizer = load_model_and_tokenizer(args.model_loc, model_type, args.lang, offline=args.offline)

    for d in ["models", model_output, log_output]:
        if not os.path.exists(d):
            os.makedirs(d)
                                
    model.config.classification_dropout = args.classifier_dropout
    print(model)
    print(f'Saving to: {model_output}')
    print("Number of Parameters:")
    print(model.num_parameters())

    # if language is multilingual, need to first pretrain on English and then fine tune as a second load in step
    data_lang = args.target_lang if args.lang == "multi" else args.lang

    if args.load_saved_dataset:
        tokenized_datasets = load_from_disk(args.dataset_loc)

    else:
        if offline:
            raw_datasets = load_from_disk(os.path.join('/home/s1948359/data/amazon_reviews_multi', data_lang))
        else:    
            raw_datasets = load_dataset('amazon_reviews_multi', data_lang)

        if args.scrub:
            print("Scrubbing data", file=sys.stderr)
            raw_datasets = raw_datasets.map(scrub)
        # Tokenize dataset, for amazon multi the sections are review_body and review_title
        tokenized_datasets = raw_datasets.map(tokenize_function_marc, batched=True,
                                              remove_columns=marc_reviews_colummns)

    # Rename fields
    # tokenized_datasets = tokenized_datasets.map(lambda examples: {'labels': examples['stars']}, batched=True)

    # format dataset object types
    if 'token_type_ids' in tokenized_datasets.column_names['train']:
        dataset_columns = ['input_ids', 'token_type_ids', 'attention_mask', 'labels']
    else:
        dataset_columns = ['input_ids', 'attention_mask', 'labels']

    tokenized_datasets.set_format(type='torch', columns=dataset_columns)
    # .map(ClassLabel

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]
    test_dataset = tokenized_datasets["test"]

    # also need to load a larger dataset with different strings, from disk location
    if not args.load_saved_dataset and args.lang == "multi" and args.target_lang == "en" and args.xl_transfer:
        # If multilingual, dataset_loc needs to be set to where the multilingual dataset lives
        us_dataset = load_from_disk(args.dataset_loc)
        tokenized_us = us_dataset.map(tokenize_function_us, batched=True,
                                      remove_columns=us_reviews_columns)
        all_datasets = [tokenized_us, tokenized_datasets]

        train_dataset = interleave_datasets([d["train"] for d in all_datasets])
        eval_dataset = interleave_datasets([d["validation"] for d in all_datasets])
        test_dataset = interleave_datasets([d["test"] for d in all_datasets])


    if args.small_test:
        train_dataset = train_dataset.shuffle(seed=args.seed).select(range(1000))
        eval_dataset = eval_dataset.shuffle(seed=args.seed).select(range(1000))
        test_dataset = test_dataset.shuffle(seed=args.seed).select(range(1000))

    print(train_dataset.column_names)

    ### Set args based on language type
    lr = args.mono_lr if not args.xl_transfer else args.transfer_lr
    steps_per_epoch = 3570 if not args.xl_transfer else 33150
    eval_every = 510

    # fine tune
    training_args = TrainingArguments(output_dir=model_output,
                                      evaluation_strategy="steps",
                                      eval_steps=eval_every,
                                      save_strategy="epoch",#"steps",
                                      #save_steps=steps_per_epoch,
                                      logging_strategy="steps",
                                      logging_steps=eval_every,
                                      logging_dir=log_output,
                                      save_total_limit=15,
                                      #load_best_model_at_end=True,
                                      per_device_train_batch_size=args.batch_size,
                                      num_train_epochs=args.epochs,
                                      seed=args.seed,
                                      weight_decay=0.01,
                                      warmup_steps=500,
                                      learning_rate=lr,
                                      report_to="wandb")
    if args.load_model:
        wandb.init(project=args.project_name, name=f"{args.lang}_{args.seed}_resume", resume=True)
        training_args.resume_from_checkpoint = model_path
    else:
        wandb.init(project=args.project_name, name=f"{lang}_{args.seed}")

    num_train_steps = ceil(len(train_dataset)/args.batch_size)
    optimiser = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_constant_schedule_with_warmup(optimiser, 500) if not args.xl_transfer else get_linear_schedule_with_warmup(optimiser, 500, num_train_steps)
    optimisers = optimiser, scheduler

    if args.lang == "ja" and args.compressed: # handling for a weird memory issue with japanese compressed models
        trainer = CustomTrainer(
            model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            optimizers=(optimisers)
        )
    else:
        trainer = Trainer(
            model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            optimizers=(optimisers)
        )

    if not args.evaluate_only:
        trainer.train()

    metrics = trainer.evaluate()
    for key, val in metrics.items():
        print('{}: {:.4f}'.format(key, val))

    wandb.finish()


