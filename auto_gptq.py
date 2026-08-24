import json
import random
from collections import defaultdict

import numpy as np
import torch

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from modelscope import AutoTokenizer


pretrained_model_dir = "meta-llama/Meta-Llama-3.1-8B-Instruct"
quantized_model_dir = "LLM-Research/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"

calibration_file = "data/quantization/calibration_wikipedia_multilingual.jsonl"

# 【修改 1】数据集共有 13 种语言，每种语言 10 条文本。
# 不再设置 nsamples=128，改为每种语言严格选 10 条，
# 最终总数为 13 × 10 = 130 条，保证各语言权重完全一致。
samples_per_language = 10
seqlen = 2048


print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_dir,
    use_fast=True
)


print("Setting random seeds...")

# 固定随机种子：后续做不同量化参数对比时，
# 每次均能得到完全相同的校准窗口。
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


print("Loading multilingual calibration dataset...")

# 【修改 2】不再将所有语言文本拼接为一条长序列。
# 按语言分别保存每篇文章的 token，确保每个 calibration sample
# 都来自单篇、单语言文本，不发生跨文章或跨语言拼接。
samples_by_language = defaultdict(list)

with open(calibration_file, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)

        text = data.get("text", "").strip()

        # 支持 JSONL 中使用 language 或 lang 作为语言字段
        language = data.get("language") or data.get("lang")

        if not text or not language:
            continue

        # 【修改 3】每篇文章独立 tokenize。
        # 不使用 truncate，先保留完整文章，后续再从文章内部随机截取。
        token_ids = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=False
        ).input_ids

        # 【修改 4】一篇文章自身必须至少有 2048 token。
        # 这样后续截取时不会连接另一篇文章，也不会连接其他语言文本。
        if token_ids.shape[1] < seqlen:
            continue

        samples_by_language[language].append(token_ids)


if not samples_by_language:
    raise ValueError(
        "没有读取到可用校准数据。请确认 JSONL 中含有 text 和 "
        "language/lang 字段，且文本长度不少于 2048 token。"
    )


print("Available calibration texts by language:")

for language, texts in sorted(samples_by_language.items()):
    print(f"  {language}: {len(texts)}")


print("Building language-balanced calibration samples...")

languages = sorted(samples_by_language.keys())

# 【修改 5】固定每种语言使用 10 条 sample。
# 不再使用“128 除以语言数”的方式，避免有些语言为 10 条、
# 有些语言为 9 条的非均衡情况。
calibration_samples = []

for language in languages:
    candidates = samples_by_language[language]

    if len(candidates) < samples_per_language:
        raise ValueError(
            f"语言 {language} 只有 {len(candidates)} 篇长度不少于 "
            f"{seqlen} token 的文本，无法抽取 "
            f"{samples_per_language} 条校准样本。"
        )

    # 【修改 6】每种语言独立抽取 10 篇文章。
    # 当前你的数据集每种语言正好 10 篇，因此每篇文章都会被使用一次。
    selected_texts = random.sample(
        candidates,
        samples_per_language
    )

    for token_ids in selected_texts:
        # 【修改 7】从当前文章内部随机截取 2048 token。
        # 每条 calibration sample 保持单语言、单文章，不会跨边界。
        max_start = token_ids.shape[1] - seqlen
        start = random.randint(0, max_start)

        input_ids = token_ids[:, start:start + seqlen]
        attention_mask = torch.ones_like(input_ids)

        calibration_samples.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask
        })


# 仅打乱样本输入顺序，不改变每种语言各 10 条的比例。
random.shuffle(calibration_samples)

expected_samples = len(languages) * samples_per_language

if len(calibration_samples) != expected_samples:
    raise RuntimeError(
        f"校准样本数异常：得到 {len(calibration_samples)}，"
        f"预期 {expected_samples}。"
    )

print(f"Calibration samples: {len(calibration_samples)}")
print(f"Languages: {languages}")
print(f"Samples per language: {samples_per_language}")
print(f"Total calibration tokens: {len(calibration_samples) * seqlen}")


quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=True,
    sym=True,
    damp_percent=0.1,
)


print("Loading unquantized model...")

model = AutoGPTQForCausalLM.from_pretrained(
    pretrained_model_dir,
    quantize_config
)


print("Quantizing...")

model.quantize(calibration_samples)


print("Saving...")

model.save_quantized(
    quantized_model_dir,
    use_safetensors=True
)

print("Done.")