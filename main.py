import os
import json
import argparse

import torch
from tqdm import tqdm

from tools import get_dataset, build_samples, load_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_type", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--quant_type", required=True)
    parser.add_argument("--model_type", required=True)
    parser.add_argument("--languages", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--save_path", default="results")

    return parser.parse_args()


@torch.no_grad()
def inference(samples, tokenizer, model, batch_size):

    records = {}
    for i in tqdm(
        range(0, len(samples), batch_size)
    ):

        batch = samples[i:i+batch_size]
        inputs = tokenizer.apply_chat_template(
            [x["messages"] for x in batch],
            add_generation_prompt=True,
            tokenize=True,
            padding=True,
            return_dict=True,
            return_tensors="pt"
        )

        outputs = model.generate(
            **inputs,
            max_new_tokens=8
        )

        lengths = inputs["attention_mask"].sum(dim=1)
        
        preds = [
            tokenizer.decode(
                output[length:],
                skip_special_tokens=True
            )
            for output, length in zip(outputs, lengths)
        ]

        for item, pred in zip(batch, preds):
            sid = item["sid"]
            if sid not in records:
                records[sid] = {
                    "question": item["question"],
                    "answer": item["answer"],
                    "predictions": []
                }
            correct = item["answer"] in pred
            records[sid]["predictions"].append(
                {
                    "template_id": item["tid"],
                    "prediction": pred,
                    "correct": correct
                }
            )

    return records


def evaluate(records):

    scores = []
    for sid, item in records.items():
        sample_acc = (
            sum(
                x["correct"]
                for x in item["predictions"]
            )
            /
            len(item["predictions"])
        )

        item["accuracy"] = sample_acc
        scores.append(sample_acc)
        
    return sum(scores) / len(scores)



def save_results(
    results,
    args
):
    save_dir = os.path.join(
        args.save_path,
        args.dataset_type,
        args.model_type,
        args.quant_type
    )

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main():

    args = parse_args()
    languages = args.languages.split(",")
    dataset_prompt, dataset_full = get_dataset(args.dataset_type)
    tokenizer, model = load_model(args)
    
    final_results = {}
    for lang in languages:

        print(f"\nEvaluating {lang}")

        samples = build_samples(dataset_full, dataset_prompt, lang)
        records = inference(
            samples,
            tokenizer,
            model,
            args.batch_size
        )

        acc = evaluate(records)
        final_results[lang] = {
            "accuracy": acc,
            "samples": len(records),
            "records": records
        }

        print(f"{lang} accuracy: {acc:.4f}")

    save_results(final_results, args)

if __name__ == "__main__":
    main()