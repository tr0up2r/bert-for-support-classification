import pandas as pd

import torch
from tqdm.notebook import tqdm
from transformers import BertTokenizer
from torch.utils.data import TensorDataset

from transformers import BertForSequenceClassification

df = pd.read_csv('data/data_for_coding.csv', encoding='ISO-8859-1')
inf_count = df['inf'].value_counts()
emt_count = df['emt'].value_counts()
print(f'inf_count :\n{inf_count}\n\nemt_count :\n{emt_count}')

