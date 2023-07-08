from torch.utils.data import Dataset
import json
import pandas as pd
import os
from PIL import Image


class SandplayDataset(Dataset):
    def __init__(self, img_dir_path, img_names_path, label_path, partition:float, transforms=None, is_train=True, is_test=False) -> None:
        super().__init__()
        self.img_dir_path = img_dir_path
        with open(img_names_path, 'r') as f:
            img_names = json.load(f)
        partition = int(partition * len(img_names["train_file"]))
        if is_train:
            self.img_names = img_names["train_file"][:partition]
        elif is_test:
            self.img_names = img_names["test_file"]
        else:
            self.img_names = img_names["train_file"][partition:]
        self.label_df = pd.read_csv(label_path)
        self.transfomrs = transforms
    
    def __len__(self): 
        return len(self.img_names)
    
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir_path, img_name, "BireView.png")
        img = Image.open(img_path).convert('RGB')
        if self.transfomrs:
            img = self.transfomrs(img)
        label = self.label_df.loc[self.label_df['name'] == img_name, 'label'].values[0]
        
        return img, label