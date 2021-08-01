import numpy as np
import pandas as pd
import torch
from transformers import BertForSequenceClassification

import emt_classifier
import inf_classifier


# Predict를 위한 inf, emt model 각각 build.
device = torch.device('cuda')
inf_model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                          num_labels=len(inf_classifier.label_dict),
                                                          output_attentions=False,
                                                          output_hidden_states=False)

inf_model.load_state_dict(torch.load('data_volume/inf_finetuned_BERT_epoch_7.model',
                                     map_location='cuda:0'))
inf_model.to(device)


emt_model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                          num_labels=len(emt_classifier.label_dict),
                                                          output_attentions=False,
                                                          output_hidden_states=False)

emt_model.load_state_dict(torch.load('data_volume/emt_finetuned_BERT_epoch_8.model',
                                     map_location=torch.device('cuda:0')))
emt_model.to(device)

inf_model.eval()
emt_model.eval()


# csv 파일 읽어와 predict 진행.
df = pd.read_csv("data/only_comments.csv")
print(df)
print(type(df))

print(f'len : {len(df)}')
comments_count = len(df)

row_list = []

with torch.no_grad():
    for i in range(comments_count):
        comment_key = df['comment_key'][i]
        comment_body = df['comment_body'][i]

        # NaN 처리를 위한 조건문.
        if type(comment_body) == float:
            continue
        link_id = df['link_id'][i]
        parent_id = df['parent_id'][i]

        inputs = emt_classifier.tokenizer(comment_body,
                                          max_length=512,
                                          padding='max_length',
                                          truncation=True,
                                          return_tensors='pt')
        inputs = inputs.to(device)

        inf_outputs = inf_model(**inputs)
        emt_outputs = emt_model(**inputs)

        inf_pred_labels = np.argmax(inf_outputs[0].cpu().detach().numpy(), axis=1)
        if inf_pred_labels == 0:
            inf_score = 1
        elif inf_pred_labels == 1:
            inf_score = 2
        else:
            inf_score = 3

        emt_pred_labels = np.argmax(emt_outputs[0].cpu().detach().numpy(), axis=1)
        if emt_pred_labels == 0:
            emt_score = 3
        elif emt_pred_labels == 1:
            emt_score = 2
        else:
            emt_score = 1

        print(i, end=' ')
        if i % 20 == 0:
            print()

        new_data = {
            'comment_key': comment_key,
            'link_id': link_id,
            'parent_id': parent_id,
            'inf_score': inf_score,
            'emt_score': emt_score
        }

        row_list.append(new_data)

        if i % 10000 == 0:
            # 결과값을 csv 파일로 저장하기 위한 dataframe 생성.
            result_df = pd.DataFrame(row_list,
                                     columns=['comment_key', 'inf_score', 'emt_score', 'link_id', 'parent_id'])
            result_df.to_csv('data/prediction_results.csv', sep=',', na_rep='NaN', index=None, mode='a', header=False)
            row_list = []

result_df = pd.DataFrame(row_list,
                         columns=['comment_key', 'inf_score', 'emt_score', 'link_id', 'parent_id'])
result_df.to_csv('data/prediction_results.csv', sep=',', na_rep='NaN', index=None, mode='a', header=False)