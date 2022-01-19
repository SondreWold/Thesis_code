import argparse
import json
import logging
from typing import List, Dict
from transformers import pipeline, Pipeline, AutoConfig, AutoTokenizer, AutoModelForMaskedLM
logger = logging.getLogger(__name__)



def evaluate_lama(model, data):
    
    correct = 0

    for line in data:
        correct = line["obj_label"]
        pred = model(line["masked_sentences"])

        logger.info(f"Sentence: {line['masked_sentences']}")
        logger.info(f"Predictions: {pred}")
    
    
    return None

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
        "--lm", type=str, help="name of the used masked language model", default="bert-base-uncased")
    parse.add_argument("--gpu", type=int, default=-1)
    parse.add_argument("--adapter_name", type=str, default=None)
    parse.add_argument("--tokenizer_name", type=str, default="bert-base-uncased")
    parse.add_argument("--use_adapter", action='store_true')

    args = parse.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.setLevel(logging.INFO)

    # Load data
    data = read_jsonl_file("../../data/LAMA/data/ConceptNet/test.jsonl")

    lm = args.lm
    config = AutoConfig.from_pretrained(lm)
    model = AutoModelForMaskedLM.from_pretrained(lm, config=config)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    device = args.gpu

    if args.use_adapter:
        logger.info("Load Adapter model")
        model.set_active_adapters([args.adapter_name])
        model.freeze_model(False)

    model = pipeline("fill-mask", model=model,
                        tokenizer=tokenizer, device=device, top_k=5)


    # Evaluate on LAMA
    accuracy = evaluate_lama(model, data)
    logger.info(accuracy)


    ###
    ### with open('data/output/predictions_lm/trex_lms_vocab/{}_{}.json'.format(pattern, lm), 'w+') as f:
    ###   json.dump(lm_results, f)
    ###

if __name__ == '__main__':
    main()
