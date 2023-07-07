import pandas as pd

label_path = "/raid/ckh/sandplay_homework/data/labels.csv"
label_df = pd.read_csv(label_path)
print(label_df.head())
print(label_df.loc[0, "name"])
print(label_df.loc[0, "label"])