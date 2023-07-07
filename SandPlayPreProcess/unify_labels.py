import json
import os
from collections import Counter
import pandas as pd

def set_label():
    dataset_path = "/raid/ckh/sandplay_homework/resource/homework_sand_label_datasets"
    file_namelist_path = "/raid/ckh/sandplay_homework/resource/new_file_namelist_infor.json"
    label2index = {
        "整合": 0, 
        "流动": 1, 
        "联结": 2, 
        "混乱": 3, 
        "分裂": 4, 
        "空洞": 5,
    }
    with open(file_namelist_path, 'r') as f:
        file_namelist = json.load(f)
    files = file_namelist["all_file"]
    # files = os.listdir(dataset_path)
    label_file_pahts = [os.path.join(dataset_path, file, 'theme_label_infor.json') for file in files]
    label_map = dict.fromkeys(files)
    for id, path in enumerate(label_file_pahts):
        with open(path, 'r') as f:
            contents = json.load(f)
        
        labels = []
        for item in contents:
            name = item["name"]
            if isinstance(name, list):
                labels.extend(name)
            elif isinstance(name, str):
                labels.append(name)
        labels_freq = Counter(labels)
        labels_freq = sorted(labels_freq.items(), key=lambda x: x[1], reverse=True)
        label_map[files[id]] = label2index[labels_freq[0][0]]
    labels_df = pd.DataFrame(label_map.items(), columns=["name", "label"])
    labels_df.to_csv("/raid/ckh/sandplay_homework/data/labels.csv", index=False)
    print("finish")

if __name__ == "__main__":
    set_label()