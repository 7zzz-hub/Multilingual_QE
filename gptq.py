import json
import torch

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer


pretrained_model_dir = "models/Meta-Llama-3.1-8B-Instruct"
quantized_model_dir = "models/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"

calibration_file = "data/calibration_wikipedia_multilingual.jsonl"


print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_dir,
    use_fast=True
)


print("Building calibration samples...")

calibration_samples = []

with open(calibration_file, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)

        text = data.get("text", "").strip()

        if not text:
            continue

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )

        # 保证长度满足GPTQ输入要求
        if inputs.input_ids.shape[1] < 2048:
            continue

        calibration_samples.append(
            {
                "input_ids": inputs.input_ids,
                "attention_mask": inputs.attention_mask
            }
        )


print(f"Calibration samples: {len(calibration_samples)}")


quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=True,
    sym=True,
    damp_percent=0.1,
)


print("Loading model...")

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