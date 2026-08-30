import json
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
import torch

LANGUAGES = ["ar","ca","el","en","es","fr","he","hu","ja","ko","nl","tr","zh"]

def get_dataset(dataset_name):
    dataset = {}
    for lang in LANGUAGES:
        with open(f"data/{dataset_name}/{lang}.json") as f:
            dataset[lang] = json.load(f)

    dataset_template = {}
    for lang in LANGUAGES:
        dataset_template[lang] = []
        for data in dataset[lang]:
            for sample in data['samples']:
                # 为每个 sample 生成所有模板变体
                tmp = []
                for template in data['prompt_templates']:
                    tmp.append({
                        "question": template.replace("<subject>", sample["subject"]).split("<mask>")[0],
                        "answer": sample["object"],
                        "index": sample["index"],
                        "subject_en": sample["subject_en"] if lang!="en" else sample["subject"],
                        "object_en": sample["object_en"] if lang!="en" else sample["object"]
                    })
                dataset_template[lang].append(tmp)
    
    return get_prompt(dataset_template)


def get_prompt(dataset, n_prompt=5):
    dataset_prompt = {}
    dataset_full = {}

    for lang in LANGUAGES:
        dataset_full[lang] = dataset[lang][n_prompt:]
        dataset_prompt[lang] = []
        for i in range(n_prompt):
            dataset_prompt[lang].extend([
                {
                    "role": "user",
                    "content": dataset[lang][i][i]["question"]
                },
                {
                    "role": "assistant",
                    "content": dataset[lang][i][i]["answer"]
                }
            ])

    return dataset_prompt, dataset_full


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
            device_map="cuda:0",
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


def build_samples(dataset_full, dataset_prompt, lang):

    samples = []
    for sid in range(len(dataset_full[lang])):
        for tid in range(len(dataset_full[lang][sid])):
            data = dataset_full[lang][sid][tid]
            samples.append({
                "sid": sid,
                "tid": tid,
                "question": data["question"],
                "answer": data["answer"],
                "messages": dataset_prompt[lang] + [{"role": "user", "content": data["question"]}]
            })
            
    return samples