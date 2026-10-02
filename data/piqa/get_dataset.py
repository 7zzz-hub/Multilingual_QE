from pathlib import Path
from datasets import load_dataset

LANGUAGE_CONFIGS = {
    "ca": "cat_latn",
    "en": "eng_latn",
    "es": "spa_latn_spai",
    "fr": "fra_latn_fran",
    "hu": "hun_latn",
    "nl": "nld_latn",
    "tr": "tur_latn",
    "ar": "arb_arab",
    "el": "ell_grek",
    "he": "heb_hebr",
    "ja": "jpn_jpan",
    "ko": "kor_hang",
    "zh": "cmn_hans",
}

output_dir = Path("global_piqa_parallel")
output_dir.mkdir(parents=True, exist_ok=True)

for lang, config in LANGUAGE_CONFIGS.items():
    dataset = load_dataset(
        "mrlbenchmarks/global-piqa-parallel",
        config,
        split="test",
    )

    output_path = output_dir / f"{lang}.jsonl"
    dataset.to_json(str(output_path), force_ascii=False)

    print(f"{lang}: {len(dataset)} questions -> {output_path}")