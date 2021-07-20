import pandas as pd
import numpy as np
import random

import torch
from tqdm.notebook import tqdm
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from transformers import BertForSequenceClassification

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score


df = pd.read_csv('data/data_for_coding_emt.csv', encoding='ISO-8859-1')
emt_count = df['emt'].value_counts()
print(f'emt_count :\n{emt_count}\n')

possible_labels = df.emt.unique()

# 3, 2, 1 역순으로 되어 있는 것을 reverse 해주어 보다 직관적인 이해를 도움.
possible_labels = possible_labels[::-1]
print(f'possible labels : {possible_labels}\n')

# label column 생성.
label_dict = {}
for index, possible_labels in enumerate(possible_labels):
    label_dict[possible_labels] = index
print(label_dict)

df['label'] = df.emt.replace(label_dict)
print(df, '\n')

# data split (train & test sets)
X_train, X_val, y_train, y_val = train_test_split(df.index.values,
                                                  df.label.values,
                                                  test_size=0.2,
                                                  random_state=42,  # seed 값 설정.
                                                  stratify=df.label.values)

df['data_type'] = ['not_set'] * df.shape[0]

df.loc[X_train, 'data_type'] = 'train'
df.loc[X_val, 'data_type'] = 'val'

print(df.groupby(['emt', 'label', 'data_type']).count())

# BERT
# uncased - upper case 문자들을 lower case로 바꾸고 난 후 tokenize한 모델.
# do_lower_case - upper case를 lower case로 변환할 것인지 여부를 정하는 param.
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                          do_lower_case=True)

encoded_data_train = tokenizer.batch_encode_plus(
    df[df.data_type == 'train'].comment.values,
    add_special_tokens=True,
    return_attention_mask=True,
    pad_to_max_length=True,
    max_length=256,
    return_tensors='pt'
)

encoded_data_val = tokenizer.batch_encode_plus(
    df[df.data_type == 'val'].comment.values,
    add_special_tokens=True,
    return_attention_mask=True,
    pad_to_max_length=True,
    max_length=256,
    return_tensors='pt'
)

input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(df[df.data_type == 'train'].label.values)

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(df[df.data_type == 'val'].label.values)

dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

# Bert 모델로는 BertForSequenceClassification 사용.
model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)

# Data Loaders
batch_size = 3

dataloader_train = DataLoader(dataset_train,
                              sampler=RandomSampler(dataset_train),
                              batch_size=batch_size)

dataloader_validation = DataLoader(dataset_val,
                                   sampler=SequentialSampler(dataset_val),
                                   batch_size=batch_size)

# Optimizer & Scheduler
# 사용할 optimizer : AdamW
optimizer = AdamW(model.parameters(),
                  lr=1e-5,  # learning rate.
                  eps=1e-8) # learning rate가 0으로 나눠지는 것을 방지하기 위한 epsilon 값.

# epochs 5로 했더니 overfitting 되는 듯.
# 3 정도가 적당?
epochs = 10

# learning rate decay를 위한 scheduler. (linear 이용)
# lr이 0부터 optimizer에서 설정한 lr까지 linear하게 warmup 됐다가 다시 0으로 linear 하게 감소.
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train) * epochs)


# Performance Metrics


def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')    # multi-class이기 때문에 weighted 사용.


def accuracy_per_class(preds, labels):
    label_dict_inverse = {v: k for k, v in label_dict.items()}

    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat == label]
        y_true = labels_flat[labels_flat == label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds == label])}/{len(y_true)}\n')


# Training Loop
device = torch.device('cpu')
seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


def evaluate(dataloader_val):
    model.eval()
    loss_val_total = 0

    predictions, true_vals = [], []

    for batch in dataloader_val:
        batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2],
                  }

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total / len(dataloader_val)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return loss_val_avg, predictions, true_vals


for epoch in tqdm(range(1, epochs + 1)):
    model.train()

    loss_train_total = 0

    progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
    i = 0
    for batch in progress_bar:
        model.zero_grad()

        # batch를 device(cpu)에 넣음.
        batch = tuple(b.to(device) for b in batch)

        # batch에서 data 추출.
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2],
                  }

        # Forward 수행.
        outputs = model(**inputs)

        # loss 구함.
        loss = outputs[0]
        print(f'i : {i}, loss : {loss}')

        # 총 loss 계산.
        loss_train_total += loss.item()
        loss.backward()

        # gradient clipping을 진행.
        # gradient exploding을 방지하기 위함으로,
        # gradient가 일정 threshold를 넘어가면 clipping을 해준다.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # gradient를 이용해 weight update.
        optimizer.step()
        # scheduler를 이용해 learning rate 조절.
        scheduler.step()

        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch))})
        i += 1

    torch.save(model.state_dict(), f'data_volume/finetuned_BERT_epoch_{epoch}_emt.model')

    tqdm.write(f'\nEpoch {epoch}')

    loss_train_avg = loss_train_total / len(dataloader_train)
    tqdm.write(f'Training loss: {loss_train_avg}')

    val_loss, predictions, true_vals = evaluate(dataloader_validation)
    val_f1 = f1_score_func(predictions, true_vals)
    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'F1 Score (Weighted): {val_f1}')