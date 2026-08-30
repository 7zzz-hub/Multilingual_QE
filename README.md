# Multilingual Quantization Evaluation

This repository compares several quantization methods on **Llama-3.1-8B-Instruct** under a multilingual evaluation setting.

## Quantization Methods

The following model precision and quantization methods are used:

| Method | Precision | Calibration Required |
|---|---:|---|
| FP16 | 16-bit | No |
| BitsAndBytes INT8 | 8-bit | No |
| BitsAndBytes NF4 | 4-bit | No |
| GPTQ INT8 | 8-bit | Yes |
| GPTQ INT4 | 4-bit | Yes |
| AWQ INT4 | 4-bit | Yes |

### FP16

The original FP16 model is used as the full-precision baseline.

### BitsAndBytes

BitsAndBytes performs quantization during model loading and does not require a separate calibration stage.

The following settings are used:

- `bnb-int8`: 8-bit quantization
- `bnb-nf4`: 4-bit NF4 quantization

### GPTQ

GPTQ is a post-training quantization method that uses calibration data.

The following settings are used:

- GPTQ INT8
- GPTQ INT4

Both GPTQ models are quantized using the multilingual Wikipedia calibration dataset.

### AWQ

AWQ is an activation-aware post-training weight quantization method.

The current experiment uses:

- AWQ INT4

AWQ also uses the multilingual Wikipedia calibration dataset.

---

# Usage

## 1. FP16

```bash
python main.py \
  --dataset_type klar \
  --checkpoint models/llama3.1-8b/llama3.1-8b-instruct \
  --quant_type fp16 \
  --model_type llama3.1 \
  --languages ar,ca,el,en,he,es,fr,hu,ja,ko,nl,tr,zh \
  --batch_size 32 \
  --save_path results
```

---

## 2. BitsAndBytes NF4

BitsAndBytes NF4 directly loads the original model in 4-bit NF4 format.

```bash
python main.py \
  --dataset_type klar \
  --checkpoint models/llama3.1-8b/llama3.1-8b-instruct \
  --quant_type bnb-nf4 \
  --model_type llama3.1 \
  --languages ar,ca,el,en,es,fr,he,hu,ja,ko,nl,tr,zh \
  --batch_size 32 \
  --save_path results
```

---

## 3. BitsAndBytes INT8

BitsAndBytes INT8 directly loads the original model in 8-bit format.

```bash
python main.py \
  --dataset_type klar \
  --checkpoint models/llama3.1-8b/llama3.1-8b-instruct \
  --quant_type bnb-int8 \
  --model_type llama3.1 \
  --languages ar,ca,el,en,es,fr,he,hu,ja,ko,nl,tr,zh \
  --batch_size 32 \
  --save_path results
```

---

## 4. GPTQ INT4

### Quantize the model

```bash
python run_gptq.py \
  --model_dir models/llama3.1-8b/llama3.1-8b-instruct \
  --save_dir models/llama3.1-8b/llama3.1-8b-gptq-int4-calib \
  --calib_file data/quantization/calibration_wikipedia_multilingual.jsonl \
  --bits 4
```

### Evaluate the quantized model

```bash
python main.py \
  --dataset_type klar \
  --checkpoint models/llama3.1-8b/llama3.1-8b-gptq-int4-calib \
  --quant_type gptq-int4 \
  --model_type llama3.1 \
  --languages ar,ca,el,en,es,fr,he,hu,ja,ko,nl,tr,zh \
  --batch_size 32 \
  --save_path results
```

---

## 5. GPTQ INT8

### Quantize the model

```bash
python run_gptq.py \
  --model_dir models/llama3.1-8b/llama3.1-8b-instruct \
  --save_dir models/llama3.1-8b/llama3.1-8b-gptq-int8-calib \
  --calib_file data/quantization/calibration_wikipedia_multilingual.jsonl \
  --bits 8
```

### Evaluate the quantized model

```bash
python main.py \
  --dataset_type klar \
  --checkpoint models/llama3.1-8b/llama3.1-8b-gptq-int8-calib \
  --quant_type gptq-int8 \
  --model_type llama3.1 \
  --languages ar,ca,el,en,es,fr,he,hu,ja,ko,nl,tr,zh \
  --batch_size 32 \
  --save_path results
```

---

## 6. AWQ INT4

### Quantize the model

```bash
python run_awq.py \
  --model_dir models/llama3.1-8b/llama3.1-8b-instruct \
  --save_dir models/llama3.1-8b/llama3.1-8b-awq-int4-calib \
  --calib_file data/quantization/calibration_wikipedia_multilingual.jsonl
```

### Evaluate the quantized model

```bash
python main.py \
  --dataset_type klar \
  --checkpoint models/llama3.1-8b/llama3.1-8b-awq-int4-calib \
  --quant_type awq-int4 \
  --model_type llama3.1 \
  --languages ar,ca,el,en,es,fr,he,hu,ja,ko,nl,tr,zh \
  --batch_size 32 \
  --save_path results
```
