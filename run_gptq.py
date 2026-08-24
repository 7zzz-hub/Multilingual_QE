import json
import random

import numpy as np
import torch

from transformers import AutoTokenizer
from gptqmodel import GPTQConfig, GPTQModel


pretrained_model_dir = "models/mistral-7b"
quantized_model_dir = (
    "models/mistral-7b-GPTQ-INT4"
)

calibration_file = (
    "data/quantization/calibration_wikipedia_multilingual.jsonl"
)

# 【修改 1】不再显式指定 ar、en、zh 等语言名称。
# 只需要知道数据集按照语言连续排列，并且每种语言的数据量相同。
num_language_groups = 13

# 每个语言组最终生成 10 条校准样本。
samples_per_language = 10

# 每条校准样本固定为 2048 token。
seqlen = 2048


print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_dir,
    use_fast=True
)


print("Setting random seeds...")

# 【修改 2】固定随机种子。
# 不同量化方法运行时，会选择相同的文章顺序和 token 片段。
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


print("Loading multilingual calibration dataset...")

# 【修改 3】读取全部记录，但不读取 language/lang 字段。
# 数据属于哪种语言，由它在文件中的位置决定。
all_records = []

with open(calibration_file, "r", encoding="utf-8") as f:
    for line_number, line in enumerate(f):
        line = line.strip()

        if not line:
            raise ValueError(
                f"JSONL 第 {line_number + 1} 行为空。"
                "空行会破坏语言分组边界。"
            )

        data = json.loads(line)
        text = data.get("text", "").strip()

        # 这里暂时保留空文本对应的位置，避免因为删除记录
        # 导致后续语言分组发生位置偏移。
        all_records.append(text)


total_records = len(all_records)

if total_records == 0:
    raise ValueError("校准数据集为空。")


# 【修改 4】总记录数必须能被 13 种语言整除。
# 例如：
#   1300 条数据 → 每个语言组 100 条；
#   13000 条数据 → 每个语言组 1000 条。
if total_records % num_language_groups != 0:
    raise ValueError(
        f"数据集共有 {total_records} 条记录，不能被 "
        f"{num_language_groups} 个语言组整除。"
    )


records_per_language = total_records // num_language_groups

print(f"Total records: {total_records}")
print(f"Language groups: {num_language_groups}")
print(f"Records per language group: {records_per_language}")


print("Building independently concatenated language samples...")

calibration_samples = []

for group_index in range(num_language_groups):
    # 【修改 5】根据行号区间切分语言组，不需要知道具体语言名称。
    #
    # 以每组 1000 条为例：
    # group 0 → 第 0～999 条
    # group 1 → 第 1000～1999 条
    # group 2 → 第 2000～2999 条
    # ...
    record_start = group_index * records_per_language
    record_end = record_start + records_per_language

    group_texts = all_records[record_start:record_end]

    # 检查当前语言组是否包含空文本。
    empty_count = sum(not text for text in group_texts)

    if empty_count > 0:
        raise ValueError(
            f"语言组 {group_index} 中存在 {empty_count} 条空文本。"
            "请先清理数据，同时保持每种语言的记录数一致。"
        )

    # 【修改 6】只在当前语言组内部打乱文本顺序。
    # 不会将不同语言组的文章混合在一起。
    random.shuffle(group_texts)

    group_token_ids = []

    for text in group_texts:
        article_token_ids = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False
        )["input_ids"]

        if not article_token_ids:
            continue

        group_token_ids.extend(article_token_ids)

        # 在两篇文章之间加入 EOS，保留文章边界。
        if tokenizer.eos_token_id is not None:
            group_token_ids.append(tokenizer.eos_token_id)

    total_group_tokens = len(group_token_ids)

    # 【修改 7】每个语言组至少需要：
    # 10 × 2048 = 20480 token。
    required_tokens = samples_per_language * seqlen

    if total_group_tokens < required_tokens:
        raise ValueError(
            f"语言组 {group_index} 只有 {total_group_tokens} token，"
            f"至少需要 {required_tokens} token，才能生成 "
            f"{samples_per_language} 条长度为 {seqlen} 的样本。"
        )

    group_token_ids = torch.tensor(
        [group_token_ids],
        dtype=torch.long
    )

    # 【修改 8】把当前语言的长 token 序列划分为多个
    # 不重叠的 2048-token 候选片段。
    available_segments = total_group_tokens // seqlen

    # 从候选片段中随机选取 10 条。
    # random.sample 保证同一语言组内不会重复选中相同片段。
    selected_segment_ids = random.sample(
        range(available_segments),
        samples_per_language
    )

    for segment_id in selected_segment_ids:
        start = segment_id * seqlen
        end = start + seqlen

        input_ids = group_token_ids[:, start:end]
        attention_mask = torch.ones_like(input_ids)

        calibration_samples.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask
        })

    print(
        f"Language group {group_index}: "
        f"records={len(group_texts)}, "
        f"tokens={total_group_tokens}, "
        f"samples={samples_per_language}"
    )


# 【修改 9】最终打乱 13 个语言组的校准样本顺序。
# 只改变输入顺序，不改变每组各 10 条的样本比例。
random.shuffle(calibration_samples)

expected_samples = (
    num_language_groups * samples_per_language
)

if len(calibration_samples) != expected_samples:
    raise RuntimeError(
        f"校准样本数异常：实际得到 {len(calibration_samples)} 条，"
        f"预期 {expected_samples} 条。"
    )


print(f"Calibration samples: {len(calibration_samples)}")
print(f"Samples per language group: {samples_per_language}")
print(
    f"Total calibration tokens: "
    f"{len(calibration_samples) * seqlen}"
)


quant_config = GPTQConfig(
    bits=4,
    group_size=128,

    # 如果需要与 AutoGPTQ 的实验参数严格一致，
    # 建议取消下面三行的注释。
    # desc_act=True,
    # sym=True,
    # damp_percent=0.1,
)


print("Loading model...")

model = GPTQModel.load(
    pretrained_model_dir,
    quant_config,
    device="cuda"
)


print("Quantizing...")

# 【修改 10】最终校准数据为：
# 13 个语言组 × 每组 10 条 × 每条 2048 token
# 总计 130 条 calibration samples。
model.quantize(
    calibration_samples,
    batch_size=1
)


print("Saving...")

model.save(quantized_model_dir)

print("Done.")