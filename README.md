# Multilingual Quantization Evaluation

This repository compares several quantization methods on **Llama-3.1-8B-Instruct**, **Qwen-3-8B**, and **Gemma-2-9B-it** under a multilingual evaluation setting.

The evaluation focuses on how different post-training quantization methods affect multilingual capabilities across several tasks, including language understanding, knowledge-intensive multiple-choice question answering, and mathematical reasoning.

---

## Datasets

Three multilingual evaluation datasets are currently supported:

| Dataset | Task Type                                       | Output Format    | Main Capability                     |
| ------- | ----------------------------------------------- | ---------------- | ----------------------------------- |
| KLAR    | Multilingual language understanding / reasoning | Dataset-specific | General multilingual capability     |
| INCLUDE | Multiple-choice question answering              | A / B / C / D    | Knowledge-intensive QA              |
| MCLM    | Mathematical reasoning                          | Free-form answer | Multilingual mathematical reasoning |

### KLAR

**KLAR** is used to evaluate general multilingual language understanding and reasoning capabilities across the selected target languages.

It serves as the primary multilingual evaluation benchmark in the original experimental setup.

### INCLUDE

**INCLUDE** is a multilingual knowledge-intensive multiple-choice question answering benchmark covering a range of domains and subjects.

Each example contains one question and four candidate answers. During evaluation, the model is instructed to return the corresponding option label:

```text
A
B
C
D
```

INCLUDE is used to evaluate whether knowledge-based multilingual question answering performance is preserved after quantization.

### MCLM

**MCLM** is used to evaluate multilingual mathematical reasoning.

The dataset contains semantically aligned mathematical problems across multiple languages together with their reference answers. Unlike INCLUDE, MCLM requires the model to directly generate the final mathematical answer rather than select from predefined options.

For example:

```json
{
  "question": "افترض أن $\\sin D = 0.7$ ... ما هو $DE$؟",
  "answer": "\\sqrt{51}"
}
```

MCLM is used to measure whether quantization affects mathematical reasoning differently across languages.

### Languages

The current evaluation covers the following 13 languages:

```text
ar, ca, el, en, es, fr, he, hu, ja, ko, nl, tr, zh
```

where:

| Code | Language  |
| ---- | --------- |
| ar   | Arabic    |
| ca   | Catalan   |
| el   | Greek     |
| en   | English   |
| es   | Spanish   |
| fr   | French    |
| he   | Hebrew    |
| hu   | Hungarian |
| ja   | Japanese  |
| ko   | Korean    |
| nl   | Dutch     |
| tr   | Turkish   |
| zh   | Chinese   |

The exact number of available samples may differ between datasets and languages.

---

## Quantization Methods

The following model precision and quantization methods are used:

| Method            | Precision | Calibration Required |
| ----------------- | --------: | -------------------- |
| FP16              |    16-bit | No                   |
| BitsAndBytes INT8 |     8-bit | No                   |
| BitsAndBytes NF4  |     4-bit | No                   |
| GPTQ INT8         |     8-bit | Yes                  |
| GPTQ INT4         |     4-bit | Yes                  |
| AWQ INT4          |     4-bit | Yes                  |

### FP16

The original FP16 model is used as the full-precision baseline.

### BitsAndBytes

BitsAndBytes performs quantization during model loading and does not require a separate calibration stage.

The following settings are used:

* `bnb-int8`: 8-bit quantization
* `bnb-nf4`: 4-bit NF4 quantization

### GPTQ

GPTQ is a post-training quantization method that uses calibration data.

The following settings are used:

* GPTQ INT8
* GPTQ INT4

### AWQ

AWQ is an activation-aware post-training weight quantization method.

The current experiment uses:

* AWQ INT4

Both GPTQ and AWQ use the multilingual Wikipedia calibration dataset.

---

## Environment

The experiments are conducted in a Linux-based GPU environment.

| Component    | Version          |
| ------------ | ---------------- |
| Python       | 3.12             |
| PyTorch      | 2.12.1+cu130     |
| Transformers | 5.16.1           |
| GPTQ         | GPTQModel        |
| AWQ          | AutoAWQ          |
| Quantization | BitsAndBytes     |
| GPU          | NVIDIA A100 40GB |

---

# Usage

The evaluation dataset is selected through:

```bash
--dataset_type
```

Currently supported values are:

```text
klar
include
mclm
```

For example:

```bash
--dataset_type klar
```

can be replaced with:

```bash
--dataset_type include
```

or:

```bash
--dataset_type mclm
```

to evaluate the same model and quantization setting on the corresponding dataset.

---

## 1. FP16

### KLAR

```bash
python main.py \
  --dataset_type klar \
  --checkpoint models/llama3.1-8b/llama3.1-8b-instruct \
  --quant_type fp16 \
  --model_type llama3.1 \
  --languages ar,ca,el,en,es,fr,he,hu,ja,ko,nl,tr,zh \
  --batch_size 32 \
  --save_path results
```

### INCLUDE

```bash
python main.py \
  --dataset_type include \
  --checkpoint models/llama3.1-8b/llama3.1-8b-instruct \
  --quant_type fp16 \
  --model_type llama3.1 \
  --languages ar,ca,el,en,es,fr,he,hu,ja,ko,nl,tr,zh \
  --batch_size 32 \
  --save_path results
```

### MCLM

```bash
python main.py \
  --dataset_type mclm \
  --checkpoint models/llama3.1-8b/llama3.1-8b-instruct \
  --quant_type fp16 \
  --model_type llama3.1 \
  --languages ar,ca,el,en,es,fr,he,hu,ja,ko,nl,tr,zh \
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

To evaluate INCLUDE or MCLM, replace:

```text
--dataset_type klar
```

with:

```text
--dataset_type include
```

or:

```text
--dataset_type mclm
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

For all quantization methods, `--dataset_type` can be switched among `klar`, `include`, and `mclm` without changing the quantized model.
