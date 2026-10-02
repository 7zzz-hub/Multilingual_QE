import json


LABELS = ["A", "B", "C", "D"]


class PIQADataset:
    def __init__(self, data_dir="data/piqa", languages=None):
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
                    f"A. {sample['solution0']}",
                    f"B. {sample['solution1']}",
                    f"C. {sample['solution2']}",
                    f"D. {sample['solution3']}",
                    f"Please output only the correct option letter: A, B, C, or D."
                ])

                question = (
                    f"Question:\n{sample['prompt']}\n\n"
                    f"Choices:\n{choices}\n\n"
                    f"Answer:"
                )

                # 0 → A, 1 → B, 2 → C, 3 → D
                answer = LABELS[int(sample["label"])-1]

                dataset_full[lang].append([{
                    "question": question,
                    "answer": answer,
                    "index": sample['example_id'],
                    "categories": sample['categories']
                }])

        return dataset_full