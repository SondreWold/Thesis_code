import argparse
import json
import logging
import random
from tqdm import tqdm
from typing import List, Dict
from transformers import pipeline, Pipeline, AutoConfig, AutoTokenizer, AutoModelForMaskedLM
logger = logging.getLogger(__name__)
from datasets import load_dataset



def evaluate_lama(model, data, at_k, is_logging=False):
    '''
    Calculates the precision @ k for a model on the LAMA dataset. If k=1, then we get normal accuracy.
    Since we have only one relevant item irrespective of k, p@k=1 if the term is in the top k documents. 
    '''
    points = 0
    n = len(data)
    for line in tqdm(data):
        correct = line["obj_label"]
        # Ignore OOV words. 
        obj_label_id = model.tokenizer.vocab.get(correct)
        if obj_label_id is None:
            n -= 1
            continue
        if is_logging: logger.info(f"Correct answer is {correct}")
        predictions = model(line["masked_sentence"])
        for pred in predictions[0:at_k]:
            if is_logging: logger.info(f"Prediction was {pred['token_str']}")
            if pred["token_str"] == correct:
                points += 1
    return points/n

def read_jsonl_file(filename: str) -> List[Dict]:
    dataset = []
    with open(filename) as f:
        for line in f:
            loaded_example = json.loads(line)
            dataset.append(loaded_example)

    return dataset

def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument(
        "--model_name_or_path", type=str, help="name of the used masked language model", default="bert-base-uncased")
    parse.add_argument("--gpu", type=int, default=-1)
    parse.add_argument("--lama_path", type=str, default=None)
    parse.add_argument("--at_k", type=int, default=1)
    parse.add_argument("--adapter_name", type=str, default=None)
    parse.add_argument("--tokenizer_name", type=str, default="bert-base-uncased")
    parse.add_argument("--use_adapter", action='store_true')
    parse.add_argument('--full_eval', action='store_true')

    args = parse.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.setLevel(logging.INFO)

    data = load_dataset("lama", "conceptnet")["train"] #from Huggingface

    lm = args.model_name_or_path
    logging.info(f"Initializing a model from name or path: {lm} and tokenizer {args.tokenizer_name}")
    config = AutoConfig.from_pretrained(lm)
    base_model = AutoModelForMaskedLM.from_pretrained(lm, config=config)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    device = args.gpu

    name = lm if not args.use_adapter else args.adapter_name
    adapter_flag = "adapter" if args.use_adapter else "normal"
    if args.use_adapter:
        logger.info("Load Adapter model")
        base_model.set_active_adapters([args.adapter_name])
        base_model.freeze_model(False)

    if args.full_eval:
        results = {}
        for k in [1,10,100]:
            logging.info(f"Calculating for k={k}")
            model = pipeline("fill-mask", model=base_model,
                        tokenizer=tokenizer, device=device, top_k=k)
            mean_p_at_k = evaluate_lama(model, data, k)
            logger.info(f"Precision for model @{k} was {mean_p_at_k}")
            results[k] = mean_p_at_k
        with open(f"./lama_results_{adapter_flag}_{name}_{args.tokenizer_name}_.txt", 'w+') as f:
            f.write(f"Results for model loaded from path {args.model_name_or_path} with tokenizer: {args.tokenizer_name}")
            for key, value in results.items():
                f.write(f"Precision@{key}: {value} \n")
                
    else:
        model = pipeline("fill-mask", model=base_model,
                        tokenizer=tokenizer, device=device, top_k=args.at_k)
        mean_p_at_k = evaluate_lama(model, data, args.at_k)
        logger.info(f"Precision for model @{args.at_k} was {mean_p_at_k}")

if __name__ == '__main__':
    main()
