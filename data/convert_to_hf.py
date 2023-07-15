import json
from tabulate import tabulate



def load_data(filepath):
    data = []
    with open(filepath) as f:
        for line in f.read().splitlines():
            instance = {}
            item = json.loads(line)
            instance["id"] = item["id"]
            prefix = "SCENE DESCRIPTION:"
            if prefix not in item["input"]:
                continue
            index = item["input"].index(prefix)
            instance["input_task"] = item["input"][:index].strip()
            instance["input_table"] = item["input"][index+len(prefix):].strip().split("[SEP]")
            instance["input_table"] = [r.strip().split(",") for r in instance["input_table"]]
            instance["input_table_md"] = tabulate(instance["input_table"],headers='firstrow')
            instance["output_text"] = item["output"]
            instance["output_list"] = [o.split("|") for o in item["output"]]
            instance["label"] = "unseen" if "unseen" in filepath else "seen"
            data.append(instance)
    return data 

data = load_data(filepath = "data/train_data/train_table_seq_1.jsonl")
data += load_data(filepath = "data/train_data/train_table_seq_2.jsonl")
with open("data/train.jsonl", "w") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")
        