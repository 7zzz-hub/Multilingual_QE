import argparse
import json
import random
import numpy as np
import torch

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--calib_file", required=True)

    parser.add_argument("--num_languages", type=int, default=13)
    parser.add_argument("--records_per_language", type=int, default=1000)
    parser.add_argument("--samples_per_language", type=int, default=10)
    parser.add_argument("--seq_len", type=int, default=512)

    parser.add_argument("--w_bit", type=int, default=4)
    parser.add_argument("--q_group_size", type=int, default=128)
    parser.add_argument("--zero_point", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--version", default="GEMM")

    parser.add_argument("--seed", type=int, default=0)

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
        texts.append(
            json.loads(line)["text"].strip()
        )


assert len(texts) == (
    args.num_languages *
    args.records_per_language
)


# build calibration samples
calibration_data = []

for lang in range(args.num_languages):

    lang_texts = texts[
        lang * args.records_per_language:
        (lang + 1) * args.records_per_language
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


    starts = random.sample(
        range(len(ids) - args.seq_len),
        args.samples_per_language
    )


    for s in starts:
        calibration_data.append(
            ids[s:s + args.seq_len]
        )


    print(
        f"lang {lang}: {len(ids)} tokens"
    )


random.shuffle(calibration_data)

print(
    f"Calibration samples: {len(calibration_data)}"
)



# AWQ config
quant_config = {
    "zero_point": args.zero_point,
    "q_group_size": args.q_group_size,
    "w_bit": args.w_bit,
    "version": args.version,
}



# load model
model = AutoAWQForCausalLM.from_pretrained(
    args.model_dir,
    low_cpu_mem_usage=True,
    device_map="auto",
)


# quantize
model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data=calibration_data,
    max_calib_samples=len(calibration_data),
    max_calib_seq_len=args.seq_len,
    n_parallel_calib_samples=1,
)


# save
model.save_quantized(args.save_dir)
tokenizer.save_pretrained(args.save_dir)

print("Done")