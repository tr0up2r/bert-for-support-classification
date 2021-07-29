import numpy as np
import pandas as pd
import torch
from transformers import BertForSequenceClassification

import emt_classifier
import inf_classifier

# Predict를 위한 inf, emt model 각각 build.
inf_model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                          num_labels=len(inf_classifier.label_dict),
                                                          output_attentions=False,
                                                          output_hidden_states=False)
inf_model.to(inf_classifier.device)
inf_model.load_state_dict(torch.load('data_volume/inf_finetuned_BERT_epoch_7.model',
                                     map_location=torch.device('cuda')))

emt_model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                          num_labels=len(emt_classifier.label_dict),
                                                          output_attentions=False,
                                                          output_hidden_states=False)
emt_model.to(emt_classifier.device)
emt_model.load_state_dict(torch.load('data_volume/emt_finetuned_BERT_epoch_8.model',
                                     map_location=torch.device('cuda')))

inf_model.eval()
emt_model.eval()


# csv 파일 읽어와 predict 진행.
df = pd.read_csv("data/only_comments.csv")
print(df)
print(type(df))

# 최종 결과값을 csv 파일로 저장하기 위한 dataframe 생성.
result_df = pd.DataFrame(index=range(0),
                         columns=['comment_key', 'inf_score', 'emt_score'])

# 각 score에 해당하는 comment의 수를 세기 위한 dataframe 생성.
# 이는 별도의 csv 파일로 저장.
count_df = pd.DataFrame(index=range(0),
                        columns=['inf_1', 'inf_2', 'inf_3', 'emt_1', 'emt_2', 'emt_3'])

inf_count_list = [0, 0, 0]
emt_count_list = [0, 0, 0]

print(f'len : {len(df)}')
comments_count = len(df)

with torch.no_grad():
    for i in range(comments_count):
        comment_key = df['comment_key'][i]
        comment_body = df['comment_body'][i]
        inputs = emt_classifier.tokenizer(comment_body,
                                          max_length=512,
                                          padding='max_length',
                                          truncation=True,
                                          return_tensors='pt')
        inf_outputs = inf_model(**inputs)
        emt_outputs = emt_model(**inputs)

        inf_pred_labels = np.argmax(inf_outputs[0].cpu().detach().numpy(), axis=1)
        if inf_pred_labels == 0:
            inf_score = 1
            inf_count_list[0] += 1
        elif inf_pred_labels == 1:
            inf_score = 2
            inf_count_list[1] += 1
        else:
            inf_score = 3
            inf_count_list[2] += 1

        emt_pred_labels = np.argmax(emt_outputs[0].cpu().detach().numpy(), axis=1)
        if emt_pred_labels == 0:
            emt_score = 3
            emt_count_list[2] += 1
        elif emt_pred_labels == 1:
            emt_score = 2
            emt_count_list[1] += 1
        else:
            emt_score = 1
            emt_count_list[0] += 1

        print(i, end=' ')
        if i % 20 == 0:
            print()

        new_data = {
            'comment_key' : comment_key,
            'inf_score' : inf_score,
            'emt_score' : emt_score
        }

        result_df = result_df.append(new_data, ignore_index=True)


count_data = {
    'inf_1': inf_count_list[0],
    'inf_2': inf_count_list[1],
    'inf_3': inf_count_list[2],
    'emt_1': emt_count_list[0],
    'emt_2': emt_count_list[1],
    'emt_3': emt_count_list[2]
}
count_df = count_df.append(count_data, ignore_index=True)

count_df.to_csv('data/prediction_results_count.csv', sep=',', na_rep='NaN', index=None)
result_df.to_csv('data/prediction_results.csv', sep=',', na_rep='NaN', index=None)