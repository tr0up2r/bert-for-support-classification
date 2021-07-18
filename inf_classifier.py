import pandas as pd

import torch
from tqdm.notebook import tqdm
from transformers import BertTokenizer
from torch.utils.data import TensorDataset

from transformers import BertForSequenceClassification

from sklearn.model_selection import train_test_split

df = pd.read_csv('data/data_for_coding_inf.csv', encoding='ISO-8859-1')
inf_count = df['inf'].value_counts()
print(f'inf_count :\n{inf_count}\n')

possible_labels = df.inf.unique()

# 3, 2, 1 역순으로 되어 있는 것을 reverse 해주어 보다 직관적인 이해를 도움.
possible_labels = possible_labels[::-1]
print(f'possible labels : {possible_labels}\n')

# label column 생성.
label_dict = {}
for index, possible_labels in enumerate(possible_labels):
    label_dict[possible_labels] = index
print(label_dict)

df['label'] = df.inf.replace(label_dict)
print(df, '\n')


# data split (train & test sets)
X_train, X_val, y_train, y_val = train_test_split(df.index.values,
                                                  df.label.values,
                                                  test_size=0.15,
                                                  random_state=42,
                                                  stratify=df.label.values)

df['data_type'] = ['not_set']*df.shape[0]

df.loc[X_train, 'data_type'] = 'train'
df.loc[X_val, 'data_type'] = 'val'

print(df.groupby(['inf', 'label', 'data_type']).count())