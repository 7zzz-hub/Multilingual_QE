import json

class KLARDataset:
    def __init__(self, data_dir="data/klar", languages=None):
        self.data_dir = data_dir
        self.languages = languages

    def get_prompt(self, dataset, n_prompt=5):
        dataset_prompt, dataset_full = {}, {}

        for lang in self.languages:
            dataset_full[lang] = dataset[lang][n_prompt:]
            dataset_prompt[lang] = []

            for i in range(n_prompt):
                sample = dataset[lang][i][i]
                dataset_prompt[lang].extend([
                    {"role": "user", "content": sample["question"]},
                    {"role": "assistant", "content": sample["answer"]}
                ])

        return dataset_full, dataset_prompt

    def load(self):
        dataset_template = {}

        for lang in self.languages:
            with open(f"{self.data_dir}/{lang}.json", "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            dataset_template[lang] = []

            for data in raw_data:
                for sample in data["samples"]:
                    variants = []

                    for template in data["prompt_templates"]:
                        variants.append({
                            "question": template.replace(
                                "<subject>", sample["subject"]
                            ).split("<mask>")[0],
                            "answer": sample["object"],
                            "index": sample["index"],
                            "subject_en": sample["subject"] if lang == "en" else sample["subject_en"],
                            "object_en": sample["object"] if lang == "en" else sample["object_en"]
                        })

                    dataset_template[lang].append(variants)

        return self.get_prompt(dataset_template)