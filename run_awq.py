import json
import random

import numpy as np
import torch

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer


pretrained_model_dir = "models/qwen3-8b"
quantized_model_dir = "models/qwen3-8b-AWQ-INT4"

calibration_file = (
    "data/quantization/calibration_wikipedia_multilingual.jsonl"
)


# 【修改 1】数据按照语言连续排列，共有 13 个语言组。
# 不需要 JSONL 显式提供 language/lang 字段。
num_language_groups = 13

# 保持每个语言组 10 条，最终共 13 × 10 = 130 条。
samples_per_language = 10

# 【修改 2】不再照搬 GPTQ 的 2048 token。
# 使用 AutoAWQ 默认采用的 512-token 校准长度。
calib_seq_len = 512


print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_dir,
    use_fast=True
)


print("Setting random seeds...")

# 【修改 3】固定随机种子，保证重复实验时使用相同校准样本。
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)


print("Loading multilingual calibration dataset...")


# 【修改 4】读取全部文本，不读取 language/lang 字段。
# 语言分组由文本在 JSONL 文件中的位置确定。
all_records = []

with open(calibration_file, "r", encoding="utf-8") as f:
    for line_number, line in enumerate(f):
        line = line.strip()

        # 不能直接跳过空行，否则可能改变语言组的位置边界。
        if not line:
            raise ValueError(
                f"JSONL 第 {line_number + 1} 行为空，"
                "会破坏语言组边界。"
            )

        data = json.loads(line)
        text = data.get("text", "").strip()

        # 暂时保留空文本的位置，分组后再检查。
        all_records.append(text)


total_records = len(all_records)

if total_records == 0:
    raise ValueError("校准数据集为空。")


# 【修改 5】每个语言组的数据量必须相同。
if total_records % num_language_groups != 0:
    raise ValueError(
        f"数据集共有 {total_records} 条记录，不能被 "
        f"{num_language_groups} 个语言组整除。"
    )


records_per_language = total_records // num_language_groups
expected_samples = num_language_groups * samples_per_language

print(f"Total records: {total_records}")
print(f"Language groups: {num_language_groups}")
print(f"Records per language group: {records_per_language}")


print("Building AWQ calibration samples...")


# 【修改 6】AutoAWQ 支持直接接收 List[List[int]]。
# 每个元素是一条长度为 512 的 token ID 序列。
#
# 这样可以避免：
# 1. AutoAWQ 对字符串再次 tokenize；
# 2. 不同语言的文本被重新拼接到同一条 sample；
# 3. 过长原始文章被 AutoAWQ 直接过滤。
calibration_data = []


for group_index in range(num_language_groups):

    # 根据数据位置切分语言组。
    record_start = group_index * records_per_language
    record_end = record_start + records_per_language

    group_texts = all_records[record_start:record_end]


    empty_count = sum(not text for text in group_texts)

    if empty_count > 0:
        raise ValueError(
            f"语言组 {group_index} 中存在 {empty_count} 条空文本。"
            "请先清理数据，并保持各语言组记录数一致。"
        )


    # 【修改 7】仅打乱当前语言组内部的文章顺序。
    # 不会混合不同语言组。
    random.shuffle(group_texts)


    # 将当前语言组独立拼接成一个长 token 序列。
    group_token_ids = []

    for text in group_texts:
        article_token_ids = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False
        )["input_ids"]

        if not article_token_ids:
            raise ValueError(
                f"语言组 {group_index} 中存在 tokenize 后为空的文本。"
            )

        group_token_ids.extend(article_token_ids)

        # 同语言的不同文章之间添加 EOS，保留文章边界。
        if tokenizer.eos_token_id is not None:
            group_token_ids.append(tokenizer.eos_token_id)


    total_group_tokens = len(group_token_ids)
    required_tokens = samples_per_language * calib_seq_len

    if total_group_tokens < required_tokens:
        raise ValueError(
            f"语言组 {group_index} 只有 {total_group_tokens} token，"
            f"至少需要 {required_tokens} token，才能生成 "
            f"{samples_per_language} 条 {calib_seq_len}-token 样本。"
        )


    # 【修改 8】按照 AWQ 的校准长度，切分为互不重叠的
    # 512-token 候选片段。
    available_segments = total_group_tokens // calib_seq_len

    if available_segments < samples_per_language:
        raise ValueError(
            f"语言组 {group_index} 只有 "
            f"{available_segments} 个完整候选片段。"
        )


    # 每个语言组随机选择 10 个不重复片段。
    selected_segment_ids = random.sample(
        range(available_segments),
        samples_per_language
    )


    for segment_id in selected_segment_ids:
        start = segment_id * calib_seq_len
        end = start + calib_seq_len

        input_ids = group_token_ids[start:end]

        if len(input_ids) != calib_seq_len:
            raise RuntimeError(
                f"校准样本长度异常：得到 {len(input_ids)}，"
                f"预期 {calib_seq_len}。"
            )

        calibration_data.append(input_ids)


    print(
        f"Language group {group_index}: "
        f"records={len(group_texts)}, "
        f"tokens={total_group_tokens}, "
        f"available_segments={available_segments}, "
        f"selected_samples={samples_per_language}"
    )


# 【修改 9】打乱不同语言组样本的输入顺序。
# 只改变顺序，不改变每组各 10 条的比例。
random.shuffle(calibration_data)


if len(calibration_data) != expected_samples:
    raise RuntimeError(
        f"校准样本数异常：实际 {len(calibration_data)}，"
        f"预期 {expected_samples}。"
    )


if any(
    len(sample) != calib_seq_len
    for sample in calibration_data
):
    raise RuntimeError(
        f"存在长度不等于 {calib_seq_len} token 的校准样本。"
    )


print(f"Calibration samples: {len(calibration_data)}")
print(f"Samples per language group: {samples_per_language}")
print(f"Tokens per sample: {calib_seq_len}")
print(
    f"Total calibration tokens: "
    f"{len(calibration_data) * calib_seq_len}"
)


quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM",
}


print("Loading model...")

model = AutoAWQForCausalLM.from_pretrained(
    pretrained_model_dir,
    low_cpu_mem_usage=True,
    device_map="auto",
)


print("Quantizing...")


# 【修改 10】明确告诉 AutoAWQ：
# 1. 使用全部 130 条，而不是默认最多 128 条；
# 2. 每条最大长度为 512 token；
# 3. 每次并行处理 1 条以降低显存占用。
#
# n_parallel_calib_samples 只影响显存和速度，
# 不改变校准样本内容。
model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data=calibration_data,
    max_calib_samples=expected_samples,
    max_calib_seq_len=calib_seq_len,
    n_parallel_calib_samples=1
)


print("Saving...")

model.save_quantized(
    quantized_model_dir
)

tokenizer.save_pretrained(
    quantized_model_dir
)

print("Done.")