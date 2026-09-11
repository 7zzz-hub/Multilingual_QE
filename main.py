import os
import json
import argparse
import math

import torch
import torch.nn.functional as F
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


def render_generation_prompt(messages, tokenizer, model_type):
    """Render the prompt ending at the assistant-generation marker."""
    kwargs = {
        "add_generation_prompt": True,
        "tokenize": False,
    }
    if model_type == "qwen3":
        # PPL is defined for an immediate answer, without a thinking section.
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(messages, **kwargs)


@torch.no_grad()
def score_continuations(prompts, targets, tokenizer, model, batch_size):
    """Return answer-only NLL, token count and PPL for each continuation."""
    scores = []
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"

    try:
        for start in tqdm(
            range(0, len(prompts), batch_size),
            desc="Calculating PPL"
        ):
            batch_prompts = prompts[start:start + batch_size]
            batch_targets = targets[start:start + batch_size]
            prompt_ids_list = []
            full_ids_list = []

            for prompt, target in zip(batch_prompts, batch_targets):
                prompt_ids = tokenizer(
                    prompt,
                    add_special_tokens=False
                )["input_ids"]
                full_ids = tokenizer(
                    prompt + target,
                    add_special_tokens=False
                )["input_ids"]

                if full_ids[:len(prompt_ids)] != prompt_ids:
                    raise ValueError(
                        "The prompt/answer tokenization boundary changed. "
                        "Please inspect the model chat template."
                    )
                if len(full_ids) == len(prompt_ids):
                    raise ValueError("The PPL target is empty after tokenization.")

                prompt_ids_list.append(prompt_ids)
                full_ids_list.append(full_ids)

            max_length = max(len(ids) for ids in full_ids_list)
            input_ids = []
            attention_mask = []
            labels = []

            for prompt_ids, full_ids in zip(prompt_ids_list, full_ids_list):
                padding_length = max_length - len(full_ids)
                input_ids.append(
                    full_ids + [tokenizer.pad_token_id] * padding_length
                )
                attention_mask.append(
                    [1] * len(full_ids) + [0] * padding_length
                )
                # Prompt and padding are conditions only; PPL covers answer tokens.
                labels.append(
                    [-100] * len(prompt_ids)
                    + full_ids[len(prompt_ids):]
                    + [-100] * padding_length
                )

            input_ids = torch.tensor(input_ids, device=model.device)
            attention_mask = torch.tensor(attention_mask, device=model.device)
            labels = torch.tensor(labels, device=model.device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            shift_logits = outputs.logits[:, :-1, :].float()
            shift_labels = labels[:, 1:]
            valid_mask = shift_labels.ne(-100)

            token_nll = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=-100,
                reduction="none"
            ).view_as(shift_labels)

            nll_sums = (token_nll * valid_mask).sum(dim=1)
            token_counts = valid_mask.sum(dim=1)

            for nll_sum, token_count in zip(nll_sums, token_counts):
                nll = nll_sum.item()
                count = token_count.item()
                mean_nll = nll / count
                scores.append({
                    "nll": nll,
                    "num_tokens": count,
                    "mean_nll": mean_nll,
                    "ppl": math.exp(mean_nll),
                })
    finally:
        tokenizer.padding_side = old_padding_side

    return scores


def prediction_lookup(records):
    return {
        (sid, prediction["template_id"]): prediction
        for sid, record in records.items()
        for prediction in record["predictions"]
    }


def calculate_klar_ppl(samples, records, tokenizer, model, model_type, batch_size):
    prompts = [
        render_generation_prompt(sample["messages"], tokenizer, model_type)
        for sample in samples
    ]
    targets = [str(sample["answer"]) for sample in samples]
    scores = score_continuations(
        prompts, targets, tokenizer, model, batch_size
    )

    lookup = prediction_lookup(records)
    for sample, score in zip(samples, scores):
        lookup[(sample["sid"], sample["tid"])].update({
            "answer_nll": score["nll"],
            "answer_tokens": score["num_tokens"],
            "answer_mean_nll": score["mean_nll"],
            "answer_ppl": score["ppl"],
        })

    total_nll = sum(score["nll"] for score in scores)
    total_tokens = sum(score["num_tokens"] for score in scores)
    return {
        "ppl_type": "answer_token_corpus_ppl",
        "ppl": math.exp(total_nll / total_tokens),
        "total_nll": total_nll,
        "total_answer_tokens": total_tokens,
        "instances": len(scores),
    }


def normalize_include_answer(answer):
    answer = str(answer).strip().upper()
    if answer and answer[0] in "ABCD":
        return answer[0]
    raise ValueError(f"Invalid INCLUDE answer: {answer!r}")


@torch.no_grad()
def inference_include(samples, tokenizer, model, model_type, batch_size):
    """Use one prompt forward pass for INCLUDE prediction and choice-PPL."""
    choices = ["A", "B", "C", "D"]
    records = {}
    total_choice_nll = 0.0

    for start in tqdm(
        range(0, len(samples), batch_size),
        desc="Evaluating INCLUDE"
    ):
        batch = samples[start:start + batch_size]
        chat_template_kwargs = {
            "add_generation_prompt": True,
            "tokenize": True,
            "padding": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        if model_type == "qwen3":
            chat_template_kwargs["enable_thinking"] = False

        inputs = tokenizer.apply_chat_template(
            [sample["messages"] for sample in batch],
            **chat_template_kwargs
        ).to(model.device)
        outputs = model(**inputs)

        # Inputs are left padded, so the last position is the next-token
        # prediction position for every prompt in the batch.
        next_token_logits = outputs.logits[:, -1, :].float()
        vocabulary_logprobs = F.log_softmax(next_token_logits, dim=-1)

        for row, sample in enumerate(batch):
            prompt = render_generation_prompt(
                sample["messages"], tokenizer, model_type
            )
            prompt_ids = tokenizer(
                prompt,
                add_special_tokens=False
            )["input_ids"]
            actual_prompt_ids = inputs["input_ids"][row][
                inputs["attention_mask"][row].bool()
            ].tolist()

            if actual_prompt_ids != prompt_ids:
                raise ValueError(
                    "Rendered INCLUDE prompt does not match model input IDs."
                )

            choice_token_ids = []
            for choice in choices:
                full_ids = tokenizer(
                    prompt + choice,
                    add_special_tokens=False
                )["input_ids"]
                if full_ids[:len(prompt_ids)] != prompt_ids:
                    raise ValueError(
                        f"Tokenization boundary changed for INCLUDE choice {choice}."
                    )
                continuation_ids = full_ids[len(prompt_ids):]
                if len(continuation_ids) != 1:
                    raise ValueError(
                        f"INCLUDE choice {choice!r} has {len(continuation_ids)} "
                        "tokens in this context; the one-forward optimization "
                        "requires exactly one token per choice."
                    )
                choice_token_ids.append(continuation_ids[0])

            raw_logprobs = vocabulary_logprobs[row, choice_token_ids]
            normalized_logprobs = F.log_softmax(
                next_token_logits[row, choice_token_ids], dim=0
            )

            gold = normalize_include_answer(sample["answer"])
            gold_index = choices.index(gold)
            prediction_index = normalized_logprobs.argmax().item()
            prediction = choices[prediction_index]
            choice_nll = -normalized_logprobs[gold_index].item()
            total_choice_nll += choice_nll

            sid = sample["sid"]
            if sid not in records:
                records[sid] = {
                    "question": sample["question"],
                    "answer": sample["answer"],
                    "predictions": []
                }
            records[sid]["predictions"].append({
                "template_id": sample["tid"],
                "prediction": prediction,
                "correct": prediction == gold,
                "choice_nll": choice_nll,
                "choice_ppl": math.exp(choice_nll),
                "choice_probability": math.exp(-choice_nll),
                "candidate_logprobs": {
                    choice: raw_logprobs[index].item()
                    for index, choice in enumerate(choices)
                },
                "candidate_probabilities": {
                    choice: normalized_logprobs[index].exp().item()
                    for index, choice in enumerate(choices)
                },
            })

    mean_choice_nll = total_choice_nll / len(samples)
    ppl_result = {
        "ppl_type": "abcd_normalized_choice_ppl",
        "ppl": math.exp(mean_choice_nll),
        "mean_choice_nll": mean_choice_nll,
        "instances": len(samples),
        "single_forward_per_prompt": True,
    }
    return records, ppl_result


def calculate_ppl(
    dataset_type,
    samples,
    records,
    tokenizer,
    model,
    model_type,
    batch_size
):
    if dataset_type == "klar":
        return calculate_klar_ppl(
            samples, records, tokenizer, model, model_type, batch_size
        )
    return None


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
        if args.dataset_type == "include":
            records, ppl_result = inference_include(
                samples,
                tokenizer,
                model,
                args.model_type,
                args.batch_size
            )
        else:
            records = inference(
                samples,
                tokenizer,
                model,
                args.model_type,
                args.enable_thinking,
                args.batch_size,
                args.max_new_tokens
            )
            ppl_result = calculate_ppl(
                args.dataset_type,
                samples,
                records,
                tokenizer,
                model,
                args.model_type,
                args.batch_size
            )

        acc = evaluate(records)
        final_results[lang] = {
            "accuracy": acc,
            "samples": len(records),
            "records": records
        }

        if ppl_result is not None:
            final_results[lang]["ppl"] = ppl_result["ppl"]
            final_results[lang]["ppl_metrics"] = ppl_result

        print(f"{lang} accuracy: {acc:.4f}")
        if ppl_result is not None:
            print(f"{lang} PPL: {ppl_result['ppl']:.4f}")

        save_results(final_results[lang], lang, args)
        
    save_results(final_results, "-1", args)

if __name__ == "__main__":
    main()
