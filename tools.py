import json
from dataset_loader import KLARDataset, IncludeDataset, MCLMDataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
import torch


def get_dataset(dataset_name, languages):
    
    if dataset_name == "klar":
        return KLARDataset(
            data_dir=f"data/{dataset_name}",
            languages=languages
        ).load()
        
    elif dataset_name == "include":
        dataset = IncludeDataset(
            data_dir=f"data/{dataset_name}",
            languages=languages
        ).load()
        return dataset, None
        
    elif dataset_name == "mclm":
        dataset = MCLMDataset(
            data_dir=f"data/{dataset_name}",
            languages=languages
        ).load()
        return dataset, None
        
    else:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}"
        )


def load_model(args):

    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint,
        use_fast=False
    )

    tokenizer.padding_side = "left"
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.quant_type == "fp16":
        model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    elif "bnb" in args.quant_type:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=args.quant_type == "bnb-nf4",
            load_in_8bit=args.quant_type == "bnb-int8"
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    model.eval()

    return tokenizer, model


def build_samples(dataset_full, lang, fewshot_messages=None):
    samples = []
    prefix = fewshot_messages.get(lang, []) if fewshot_messages else []

    for sid, variants in enumerate(dataset_full[lang]):
        for tid, data in enumerate(variants):
            samples.append({
                "sid": sid,
                "tid": tid,
                "question": data["question"],
                "answer": data["answer"],
                "messages": prefix + [
                    {"role": "user", "content": data["question"]}
                ]
            })

    return samples