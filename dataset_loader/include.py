import json


LABELS = ["A", "B", "C", "D"]


class IncludeDataset:
    def __init__(self, data_dir="data/include", languages=None):
        self.data_dir = data_dir
        self.languages = languages

    def load(self):
        dataset_full = {}

        for lang in self.languages:

            file_path = f"{self.data_dir}/{lang}.json"

            with open(file_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            dataset_full[lang] = []

            for i, sample in enumerate(raw_data):

                choices = "\n".join([
                    f"A. {sample['option_a']}",
                    f"B. {sample['option_b']}",
                    f"C. {sample['option_c']}",
                    f"D. {sample['option_d']}",
                    f"Please output only the correct option letter: A, B, C, or D."
                ])

                question = (
                    f"Question:\n{sample['question']}\n\n"
                    f"Choices:\n{choices}\n\n"
                    f"Answer:"
                )

                # 0 → A, 1 → B, 2 → C, 3 → D
                answer = LABELS[int(sample["answer"])]

                dataset_full[lang].append([{
                    "question": question,
                    "answer": answer,
                    "index": sample.get("index", i),
                }])

        return dataset_full