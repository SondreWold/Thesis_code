#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning a 🤗 Transformers model on multiple choice relying on the accelerate library without using a Trainer.
"""
# You can also adapt this script on your own multiple choice task. Pointers for this are left as comments.

import argparse
import logging
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import datasets
import torch
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
import transformers
from accelerate import Accelerator
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from transformers.file_utils import PaddingStrategy, get_full_repo_name


logger = logging.getLogger(__name__)
# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--tune_both",
        type=bool,
        default=False,
        help="Tune both adapter and normal weights",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float,
                        default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts",
                 "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None,
                        help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str,
                        help="The token to use to push to the Model Hub.")
    args = parser.parse_args()

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def load_data():
    ds = load_dataset('commonsense_qa')
    return ds


class InputExample(object):
    """
    A single multiple choice question.
    """

    def __init__(self, example_id, question, answers, label):
        self.example_id = example_id
        self.question = question
        self.answers = answers
        self.label = label


class InputFeatures(object):
    """
    A single feature converted from an example.
    """

    def __init__(self, example_id, choices_features, label):
        self.example_id = example_id
        self.label = label
        self.choices_features = [
            {'input_ids': input_ids, 'input_mask': input_mask,
                'segment_ids': segment_ids}
            for _, input_ids, input_mask, segment_ids in choices_features
        ]


class CommonsenseQAProcessor:
    """
    A Commonsense QA Data Processor
    """

    def __init__(self):
        self.dataset = None
        self.labels = [0, 1, 2, 3, 4]
        self.LABELS = ['A', 'B', 'C', 'D', 'E']

    def get_split(self, split='train'):
        if self.dataset is None:
            self.dataset = load_data()
        return self.dataset[split]

    def create_examples(self, split='train'):
        examples = []
        data_tr = self.get_split(split)
        example_id = 0

        for question, choices, answerKey in zip(data_tr['question'], data_tr['choices'], data_tr['answerKey']):
            answers = np.array(choices['text'])
            label = self.LABELS.index(answerKey)
            examples.append(InputExample(
                example_id=example_id, question=question,
                answers=answers, label=label
            ))
            example_id += 1

        return examples


def truncate_seq_pair(tokens_a, tokens_b, max_length):
    """
    Truncates a sequence pair in place to the maximum length.

    This is a simple heuristic which will always truncate the longer sequence one token at a time.
    This makes more sense than truncating an equal percent of tokens from each,
    since if one sequence is very short then each token that's truncated
    likely contains more information than a longer sequence.

    However, since we'd better not to remove tokens of options and questions,
    you can choose to use a bigger length or only pop from context
    """

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            warning = 'Attention! you are removing from token_b (swag task is ok). ' \
                      'If you are training ARC and RACE (you are popping question + options), ' \
                      'you need to try to use a bigger max seq length!'
            print(warning)
            tokens_b.pop()


def examples_to_features(examples, label_list, max_seq_length, tokenizer,
                         cls_token_at_end=False,
                         cls_token='[CLS]',
                         cls_token_segment_id=1,
                         sep_token='[SEP]',
                         sequence_a_segment_id=0,
                         sequence_b_segment_id=1,
                         sep_token_extra=False,
                         pad_token_segment_id=0,
                         pad_on_left=False,
                         pad_token=0,
                         mask_padding_with_zero=True):
    """
    Convert Commonsense QA examples to features.

    The convention in BERT is:
    (a) For sequence pairs:
    tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1

    (b) For single sequences:
    tokens:   [CLS] the dog is hairy . [SEP]
    type_ids:   0   0   0   0  0     0   0

    Where "type_ids" are used to indicate whether this is the first sequence or the second sequence.
    The embedding vectors for `type=0` and `type=1` were learned during pre-training
    and are added to the word piece embedding vector (and position vector).
    This is not *strictly* necessary since the [SEP] token unambiguously separates the sequences,
    but it makes it easier for the model to learn the concept of sequences.

    For classification tasks, the first vector (corresponding to [CLS]) is used as as the "sentence vector".
    Note that this only makes sense because the entire model is fine-tuned.
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in tqdm(enumerate(examples), desc="Converting examples to features", disable=True):

        choices_features = []
        for ending_idx, (question, answers) in enumerate(zip(example.question, example.answers)):

            tokens_a = tokenizer.tokenize(example.question)
            if example.question.find("_") != -1:
                tokens_b = tokenizer.tokenize(
                    example.question.replace("_", answers))
            else:
                tokens_b = tokenizer.tokenize(answers)

            special_tokens_count = 4 if sep_token_extra else 3
            truncate_seq_pair(tokens_a, tokens_b,
                              max_seq_length - special_tokens_count)

            tokens = tokens_a + [sep_token]
            if sep_token_extra:
                tokens += [sep_token]

            segment_ids = [sequence_a_segment_id] * len(tokens)

            if tokens_b:
                tokens += tokens_b + [sep_token]
                segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

            if cls_token_at_end:
                tokens = tokens + [cls_token]
                segment_ids = segment_ids + [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens.
            # Only real tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)

            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1]
                              * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] *
                               padding_length) + segment_ids

            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                input_mask = input_mask + \
                    ([0 if mask_padding_with_zero else 1] * padding_length)
                segment_ids = segment_ids + \
                    ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            choices_features.append(
                (tokens, input_ids, input_mask, segment_ids))

        label = label_map[example.label]

        if ex_index < 0:
            print("*** Example ***")
            print("race_id: {}".format(example.example_id))
            for choice_idx, (tokens, input_ids, input_mask, segment_ids) in enumerate(choices_features):
                print("choice: {}".format(choice_idx))
                print("tokens: {}".format(' '.join(tokens)))
                print("input_ids: {}".format(' '.join(map(str, input_ids))))
                print("input_mask: {}".format(' '.join(map(str, input_mask))))
                print("segment_ids: {}".format(
                    ' '.join(map(str, segment_ids))))
                print("label: {}".format(label))

        features.append(InputFeatures(
            example_id=example.example_id,
            choices_features=choices_features,
            label=label
        ))

    return features


def load_features(args, tokenizer, mode='train'):
    """
    Load the processed Commonsense QA dataset
    """

    def select_field(feature_list, field_name):
        return [
            [choice[field_name] for choice in feature.choices_features]
            for feature in feature_list
        ]

    assert mode in {'train', 'validation', 'test'}
    print("Creating features from dataset...")

    processor = CommonsenseQAProcessor()
    label_list = processor.labels
    examples = processor.create_examples(split=mode)

    print("Training number:", str(len(examples)))
    features = examples_to_features(examples, label_list, args.max_seq_length, tokenizer,
                                    cls_token_at_end=False,
                                    cls_token=tokenizer.cls_token,
                                    sep_token=tokenizer.sep_token,
                                    sep_token_extra=False,
                                    cls_token_segment_id=0,
                                    pad_on_left=False,
                                    pad_token_segment_id=0)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor(select_field(
        features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(
        features, 'input_mask'), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(
        features, 'segment_ids'), dtype=torch.long)
    all_label_ids = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask,
                            all_segment_ids, all_label_ids)
    return dataset


def main():
    args = parse_args()
    device = "cpu"

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(
        logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(
                    Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    if args.config_name:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning(
            "You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForMultipleChoice.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMultipleChoice.from_config(config)

    if args.tune_both:
        logger.info("Setting: tuning both activated")
        if "mlm" in model.config.adapters:
            logger.info("Found mlm model in adapter config")
            logger.info(f"{model.config.adapters}")
            model.train_adapter(["mlm"])  # activate adapter
            model.set_active_adapters(["mlm"])
            model.freeze_model(False)  # keep normal weights dynamic

    model.resize_token_embeddings(len(tokenizer))

    print('\n Loading training dataset')
    dataset_tr = load_features(args, tokenizer, mode='train')
    sampler_tr = RandomSampler(dataset_tr)
    train_dataloader = DataLoader(
        dataset_tr, sampler=sampler_tr, batch_size=args.batch_size)

    print('\n Loading validation dataset')
    dataset_val = load_features(args, tokenizer, 'validation')
    sampler_val = SequentialSampler(dataset_val)
    eval_dataloader = DataLoader(
        dataset_val, sampler=sampler_val, batch_size=args.batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Use the device given by the `accelerator` object.
    device = accelerator.device
    model.to(device)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(
            args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Metrics
    metric = load_metric("accuracy")

    # Train!
    total_batch_size = args.batch_size * \
        accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset_tr)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps),
                        disable=not accelerator.is_local_main_process)
    completed_steps = 0

    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels': batch[3]}
            outputs = model(**inputs)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'labels': batch[3]}
                outputs = model(**inputs)
            predictions = outputs.logits.argmax(dim=-1)
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(inputs["labels"]),
            )

        eval_metric = metric.compute()
        accelerator.print(f"epoch {epoch}: {eval_metric}")

        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, save_function=accelerator.save)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                repo.push_to_hub(
                    commit_message=f"Training in progress epoch {epoch}", blocking=False)

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training")


if __name__ == "__main__":
    main()
