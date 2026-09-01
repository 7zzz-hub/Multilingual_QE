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
    parser.add_argument("--enable_thinking", default=False)
    parser.add_argument("--max_new_tokens", type=int, required=True)
    

    return parser.parse_args()


@torch.no_grad()
def inference(samples, tokenizer, model, model_type, enable_thinking, batch_size, max_new_tokens):

    records = {}
    for i in tqdm(
        range(0, len(samples), batch_size)
    ):

        batch = samples[i:i+batch_size]

        chat_template_kwargs = {
            "add_generation_prompt": True,
            "tokenize": True,
            "padding": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        
        # enable_thinking只适用于Qwen3
        if model_type == "qwen3":
            chat_template_kwargs["enable_thinking"] = enable_thinking

        inputs = tokenizer.apply_chat_template(
            [x["messages"] for x in batch],
            **chat_template_kwargs
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens
        )
        
        input_length = inputs["input_ids"].shape[1]
        
        preds = [
            tokenizer.decode(
                output[input_length:],
                skip_special_tokens=True
            ).strip()
            for output in outputs
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
    lang,
    args
):
    save_dir = os.path.join(
        args.save_path,
        args.dataset_type,
        args.model_type,
        args.quant_type
    )

    os.makedirs(save_dir, exist_ok=True)
    
    if lang =="-1":
        with open(os.path.join(save_dir, f"{args.model_type}_{args.quant_type}_result.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    else:
        with open(os.path.join(save_dir, f"{lang}.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


def main():

    args = parse_args()
    languages = args.languages.split(",")
    dataset_full, dataset_prompt = get_dataset(args.dataset_type, languages)
    tokenizer, model = load_model(args)

    final_results = {}
    for lang in languages:

        print(f"\nEvaluating {lang}")

        samples = build_samples(dataset_full, lang, dataset_prompt)
        records = inference(
            samples,
            tokenizer,
            model,
            args.model_type,
            args.enable_thinking,
            args.batch_size,
            args.max_new_tokens
        )

        acc = evaluate(records)
        final_results[lang] = {
            "accuracy": acc,
            "samples": len(records),
            "records": records
        }

        print(f"{lang} accuracy: {acc:.4f}")

        save_results(final_results[lang], lang, args)
        
    save_results(final_results, "-1", args)

if __name__ == "__main__":
    main()