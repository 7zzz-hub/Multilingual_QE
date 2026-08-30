import argparse
import json
import random

import numpy as np
import torch

from transformers import AutoTokenizer
from gptqmodel import GPTQConfig, GPTQModel


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--calib_file", required=True)

    parser.add_argument("--num_languages", type=int, default=13)
    parser.add_argument("--samples_per_lang", type=int, default=32)
    parser.add_argument("--seqlen", type=int, default=2048)

    parser.add_argument("--bits", type=int, required=True)
    parser.add_argument("--group_size", type=int, default=128)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")

    return parser.parse_args()


args = get_args()


# seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


# tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    args.model_dir,
    use_fast=True
)


# load calibration texts
texts = []
with open(args.calib_file, "r", encoding="utf-8") as f:
    for line in f:
        t = json.loads(line)["text"].strip()
        if t:
            texts.append(t)


assert len(texts) % args.num_languages == 0

per_lang = len(texts) // args.num_languages


# build calibration samples
calib = []
for lang in range(args.num_languages):

    lang_texts = texts[
        lang * per_lang:
        (lang + 1) * per_lang
    ]

    random.shuffle(lang_texts)

    ids = []
    for t in lang_texts:
        ids.extend(
            tokenizer(
                t,
                add_special_tokens=False
            )["input_ids"]
        )

        if tokenizer.eos_token_id:
            ids.append(tokenizer.eos_token_id)


    ids = torch.tensor(
        ids,
        dtype=torch.long
    )

    n = len(ids) // args.seqlen

    assert n >= args.samples_per_lang


    for i in random.sample(
        range(n),
        args.samples_per_lang
    ):

        x = ids[
            i * args.seqlen:
            (i + 1) * args.seqlen
        ].unsqueeze(0)

        calib.append({
            "input_ids": x,
            "attention_mask": torch.ones_like(x)
        })


    print(
        f"lang {lang}: "
        f"{len(ids)} tokens"
    )


random.shuffle(calib)


print(
    f"Calibration samples: {len(calib)}"
)



# GPTQ
quant_config = GPTQConfig(
    bits=args.bits,
    group_size=args.group_size
)


model = GPTQModel.load(
    args.model_dir,
    quant_config,
    device=args.device
)


model.quantize(
    calib,
    batch_size=1
)


model.save(
    args.save_dir
)


print("Done")