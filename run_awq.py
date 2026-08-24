import json

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer


pretrained_model_dir = "models/qwen3-8b"
quantized_model_dir = "models/qwen3-8b-AWQ-INT4"

calibration_file = "data/quantization/calibration_wikipedia_multilingual.jsonl"


print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_dir,
    use_fast=True
)


print("Building calibration samples...")

calibration_data = []

with open(calibration_file, "r", encoding="utf-8") as f:

    for line in f:

        data = json.loads(line)
        text = data.get("text", "").strip()

        if not text:
            continue

        calibration_data.append(text)


print(f"Calibration samples: {len(calibration_data)}")


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


model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data=calibration_data
)


print("Saving...")

model.save_quantized(
    quantized_model_dir
)

tokenizer.save_pretrained(
    quantized_model_dir
)

print("Done.")