from datasets import load_dataset
import json

dataset = load_dataset("modelscope/self-cognition", split="train")

data = [example for example in dataset]

with open("self_cognition.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

