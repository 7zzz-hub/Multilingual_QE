import json
import torch

from transformers import AutoTokenizer
from gptqmodel import GPTQModel, QuantizeConfig


pretrained_model_dir = "models/qwen3-8b"
quantized_model_dir = "models/qwen3-8b-GPTQ-INT4"

calibration_file = "data/quantization/calibration_wikipedia_multilingual.jsonl"


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

        if inputs.input_ids.shape[1] < 2048:
            continue

        calibration_samples.append(
            {
                "input_ids": inputs.input_ids,
                "attention_mask": inputs.attention_mask
            }
        )


print(f"Calibration samples: {len(calibration_samples)}")


quantize_config = QuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=True,
    sym=True,
    damp_percent=0.1,
)


print("Loading model...")

model = GPTQModel.load(
    pretrained_model_dir,
    quantize_config=quantize_config,
    device="cuda"
)


print("Quantizing...")

model.quantize(
    calibration_samples
)


print("Saving...")

model.save(
    quantized_model_dir
)

print("Done.")