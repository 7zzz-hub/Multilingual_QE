import json


class MCLMDataset:
    def __init__(self, data_dir="data/mclm", languages=None):
        self.data_dir = data_dir
        self.languages = languages

    def load(self):
        dataset_full = {}

        for lang in self.languages:
            with open(
                f"{self.data_dir}/{lang}.json",
                "r",
                encoding="utf-8"
            ) as f:
                raw_data = json.load(f)

            dataset_full[lang] = []

            for i, sample in enumerate(raw_data):
                dataset_full[lang].append([{
                    "question": sample["question"],
                    "answer": sample["answer"],
                    "index": sample.get("index", i)
                }])

        return dataset_full