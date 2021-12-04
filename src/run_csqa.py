# Inspired by: https://github.com/Mars-tin/commonsense-for-inference/
import json
import codecs
import argparse
from copy import deepcopy
from tqdm import tqdm, trange
import random
import numpy as np
import torch
import logging
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import (AdamW, get_linear_schedule_with_warmup,
                          AutoConfig, AutoModelForMultipleChoice, AutoTokenizer)
from datasets import load_dataset

logger = logging.getLogger(__name__)


def parse_args():

    parser = argparse.ArgumentParser(
        description="Common sense question answering")
    parser.add_argument("--model_name_or_path", type=str, default=None,
                        help="Model name from HF repo or local path")

    parser.add_argument("--config_name", type=str,
                        default=None,
                        help="Pre-trained config name or path")

    parser.add_argument(
        "--tune_both",
        type=bool,
        default=False,
        help="Tune both adapter and normal weights",
    )

    parser.add_argument("--tokenizer_name", default=None, type=str,
                        help="Pre-trained tokenizer name or path if not the same as model_name")

    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. "
                        "Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--batch_size", default=3, type=int,
                        help="Batch size for training.")

    parser.add_argument("--learning_rate", default="3e-5", type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=10, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=100,
                        help="Log every n updates steps.")

    parser.add_argument('--fp16', type=bool, default=True,
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed: <int>")
    parser.add_argument(
        "--device", default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    return args


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_optimizer(args, model, train_size):
    num_training_steps = train_size // args.num_train_epochs
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps)

    return model, optimizer, scheduler


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


def train(args, model, tokenizer):

    print('\n Loading training dataset')
    dataset_tr = load_features(args, tokenizer, mode='train')
    sampler_tr = RandomSampler(dataset_tr)
    dataloader_tr = DataLoader(
        dataset_tr, sampler=sampler_tr, batch_size=args.batch_size)

    print('\n Loading validation dataset')
    dataset_val = load_features(args, tokenizer, 'validation')
    sampler_val = SequentialSampler(dataset_val)
    dataloader_val = DataLoader(
        dataset_val, sampler=sampler_val, batch_size=args.batch_size)

    model, optimizer, scheduler = load_optimizer(
        args, model, len(dataloader_tr))

    num_steps = 0
    best_steps = 0
    tr_loss = 0.0
    best_val_acc, best_val_loss = 0.0, 99999999999.0
    best_model = None
    epoch_c = 0

    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs),
                            desc="Epoch", disable=False, leave=True, position=1)

    for _ in train_iterator:

        epoch_iterator = tqdm(dataloader_tr, desc="Iteration",
                              disable=False, leave=True, position=1)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            model.zero_grad()

            batch = tuple(b.to(args.device) for b in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels': batch[3]}
            outputs = model(**inputs)
            loss = outputs[0]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()

            optimizer.step()
            scheduler.step()

            num_steps += 1

        results = evaluate(args, model, dataloader_val)
        logger.info(
            f"\n Epoch {epoch_c} evaluation: val acc:{results['val_acc']}, val loss: {results['val_loss']}")
        epoch_c += 1

    loss = tr_loss / num_steps
    return model


def evaluate(args, model, dataloader):

    val_loss = 0.0
    num_steps = 0
    preds, labels = None, None

    results = {}

    for batch in tqdm(dataloader, desc="Validation", disable=True, leave=True, position=1):
        batch = tuple(t.to(args.device) for t in batch)
        model.eval()

        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels': batch[3]}
            outputs = model(**inputs)
            loss, logits = outputs[:2]

            val_loss += loss.mean().item()

        num_steps += 1

        if preds is None:
            preds = logits.detach().cpu().numpy()
            labels = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            labels = np.append(
                labels, inputs['labels'].detach().cpu().numpy(), axis=0)

    loss = val_loss / num_steps
    preds = np.argmax(preds, axis=1)
    acc = (preds == labels).mean()
    result = {"val_acc": acc, "val_loss": loss}
    results.update(result)

    return results


def test(args, tokenizer, model):

    dataset = load_features(args, tokenizer, mode='validation')
    sampler = SequentialSampler(dataset)
    dataloader_test = DataLoader(
        dataset, sampler=sampler, batch_size=args.batch_size)

    results = evaluate(args, model, dataloader_test)
    logger.info('\nTesting...')
    logger.info("\n final validation acc: {}, final validation loss: {}"
                .format(str(results['val_acc']), str(results['val_loss'])))


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.setLevel(logging.INFO)
    args = parse_args()
    logger.info(f'Using device: {args.device}')
    set_seed(args.seed)

    processor = CommonsenseQAProcessor()
    num_labels = len(processor.labels)

    config = AutoConfig.from_pretrained(
        args.model_name_or_path, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=True)

    model = AutoModelForMultipleChoice.from_pretrained(
        args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
    model.to(args.device)

    if args.tune_both:
        logger.info("Setting: tuning both activated")
        if "mlm" in model.config.adapters:
            logger.info("Found mlm model in adapter config")
            logger.info(f"{model.config.adapters}")
            model.train_adapter(["mlm"])  # activate adapter
            model.set_active_adapters(["mlm"])
            model.freeze_model(False)  # keep normal weights dynamic

    trained_model = train(args, model, tokenizer)
    test(args, tokenizer, trained_model)


if __name__ == "__main__":
    main()
