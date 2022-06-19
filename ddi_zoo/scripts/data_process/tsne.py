import os
import pandas as pd
import re
from rdkit import Chem
from tqdm import tqdm
import numpy as np


raw_dir = 'data/new_split/inductive/new_build1/valid.csv'
valid_data = {}
valid_csv = pd.read_csv(raw_dir)
for d1, d2, nd1, nd2, y in zip(tqdm(valid_csv['id1']), valid_csv['id2'], \
    valid_csv['neg_id1'], valid_csv['neg_id2'], valid_csv['y']):

    data_dict = {
            'id1': d1,
            'id2': d2,
            'neg_id1': nd1,
            'neg_id2': nd2,
            'y': y,
        }
    if y in valid_data:
        valid_data[y].append(data_dict)
    else:
        valid_data[y] = [data_dict]

dump_data = []
for k, v in valid_data.items():
    dump_data.extend(v)

df_data = pd.DataFrame(dump_data)
df_data.to_csv("data/tsne/valid.csv")

