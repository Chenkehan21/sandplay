import json
import os
from collections import Counter
import pandas as pd

def check_label():
    sandplay_subjects = ["整合", "流动", "联结", "混乱", "分裂", "空洞"]
    dataset_path = "/raid/ckh/sandplay_homework/resource/homework_sand_label_datasets"
    dataset_split_path = "/raid/ckh/sandplay_homework/resource/file_namelist_infor.json"
    
    with open(dataset_split_path, 'r') as f:
        dataset_split = json.load(f)
    train_files = dataset_split["train_file"]
    test_files = dataset_split["test_file"]
    
    files = os.listdir(dataset_path)
    label_file_pahts = [os.path.join(dataset_path, file, 'theme_label_infor.json') for file in files]
    empty_labels = []
    all_one_vote_labels = []
    uncorrect_labels = []
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
        if len(labels) == 0:
            empty_labels.append(files[id])
        elif labels_freq[0][1] == 1:
            all_one_vote_labels.append(files[id])
        elif labels_freq[0][0] not in sandplay_subjects:
            uncorrect_labels.append(files[id])
            print(labels_freq)
    
    intersection = set(empty_labels) & set(all_one_vote_labels) & set(uncorrect_labels)
    union = list(set(empty_labels) | set(all_one_vote_labels) | set(uncorrect_labels))
    updated_train_files = [file for file in train_files if file not in union]
    updated_test_files = [file for file in test_files if file not in union]
    new_file_namelist = {
        "all_file": updated_train_files + updated_test_files,
        "train_file": updated_train_files,
        "test_file": updated_test_files
    }
    with open("/raid/ckh/sandplay_homework/resource/new_file_namelist_infor.json", 'w') as f:
        json.dump(new_file_namelist, f)
    
    print("\n\nempty_labels: ", empty_labels, len(empty_labels))
    print("\n\nall one vote: ", all_one_vote_labels, len(all_one_vote_labels))
    print("\n\nuncorrect_labels: ", uncorrect_labels, len(uncorrect_labels))
    print("\n\nintersection: ", list(intersection))
    print("\n\nnumber of all samples: ", len(files))
    print("nnumber of noise samples: ", len(empty_labels) + len(all_one_vote_labels) + len(uncorrect_labels))
    print("number of raw train samples: ", len(train_files))
    print("number of raw test samples: ", len(test_files))
    print("number of updated train samples: ", len(updated_train_files))
    print("number of updated test samples: ", len(updated_test_files))
    
    
if __name__ == "__main__":
    check_label()