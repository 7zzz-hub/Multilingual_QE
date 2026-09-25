import json


LABELS = ["A", "B", "C", "D"]


class BelebeleDataset:
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
                    f"A. {sample['mc_answer1']}",
                    f"B. {sample['mc_answer2']}",
                    f"C. {sample['mc_answer3']}",
                    f"D. {sample['mc_answer4']}",
                    f"Please output only the correct option letter: A, B, C, or D."
                ])

                question = (
                    f"Passage:\n{sample['flores_passage']}\n\n"
                    f"Question:\n{sample['question']}\n\n"
                    f"Choices:\n{choices}\n\n"
                    f"Answer:"
                )

                # 0 → A, 1 → B, 2 → C, 3 → D
                answer = LABELS[int(sample["correct_answer_num"])-1]

                dataset_full[lang].append([{
                    "question": question,
                    "answer": answer,
                    "index": sample.get("question_number", i),
                }])

        return dataset_full