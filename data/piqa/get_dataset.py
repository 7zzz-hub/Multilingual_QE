# from pathlib import Path
# from datasets import load_dataset

# LANGUAGE_CONFIGS = {
#     "ca": "cat_latn",
#     "en": "eng_latn",
#     "es": "spa_latn_spai",
#     "fr": "fra_latn_fran",
#     "hu": "hun_latn",
#     "nl": "nld_latn",
#     "tr": "tur_latn",
#     "ar": "arb_arab",
#     "el": "ell_grek",
#     "he": "heb_hebr",
#     "ja": "jpn_jpan",
#     "ko": "kor_hang",
#     "zh": "cmn_hans",
# }

# output_dir = Path("global_piqa_parallel")
# output_dir.mkdir(parents=True, exist_ok=True)

# for lang, config in LANGUAGE_CONFIGS.items():
#     dataset = load_dataset(
#         "mrlbenchmarks/global-piqa-parallel",
#         config,
#         split="test",
#     )

#     output_path = output_dir / f"{lang}.jsonl"
#     dataset.to_json(str(output_path), force_ascii=False)

#     print(f"{lang}: {len(dataset)} questions -> {output_path}")

import json
from pathlib import Path

# 当前目录
cwd = Path(".")

# 找所有 .jsonl 文件（不递归子目录）
jsonl_files = sorted(cwd.glob("*.jsonl"))

if not jsonl_files:
    print("当前目录下没有找到 .jsonl 文件")
else:
    for src in jsonl_files:
        dst = src.with_suffix(".json")   # 同名，后缀换成 .json

        # 流式写入，省内存，适合大文件
        with open(src, "r", encoding="utf-8") as fin, \
             open(dst, "w", encoding="utf-8") as fout:
            fout.write("[\n")
            first = True
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                if not first:
                    fout.write(",\n")
                fout.write(json.dumps(json.loads(line), ensure_ascii=False))
                first = False
            fout.write("\n]\n")

        print(f"已转换: {src.name} -> {dst.name}")

print("全部完成")