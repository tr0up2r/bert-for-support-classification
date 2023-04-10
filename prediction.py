import os.path

import numpy as np
import pandas as pd
import torch
from transformers import BertForSequenceClassification
from tqdm import tqdm

import emt_classifier
import inf_classifier
import classifier

device = torch.device('cuda')


def predict():
    # load models
    inf_model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                              num_labels=len(inf_classifier.label_dict),
                                                              output_attentions=False,
                                                              output_hidden_states=False)

    inf_model.load_state_dict(torch.load('models/inf_finetuned_BERT_epoch_7.model',
                                         map_location='cuda:0'))
    inf_model.to(device)

    emt_model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                              num_labels=len(emt_classifier.label_dict),
                                                              output_attentions=False,
                                                              output_hidden_states=False)

    emt_model.load_state_dict(torch.load('models/emt_finetuned_BERT_epoch_8.model',
                                         map_location=torch.device('cuda:0')))
    emt_model.to(device)

    # evaluation mode
    inf_model.eval()
    emt_model.eval()

    # read csv
    # columns of dataset: ['comment_key', 'comment_body']
    df = pd.read_csv("data/only_comments.csv")  # the path of your dataset file want to predict
    comments_count = len(df)

    row_list = []
    columns = ['comment_key', 'inf_score', 'emt_score']  # your own column names
    path = 'prediction_results.csv'  # your own file path

    with torch.no_grad():
        for i in tqdm(range(comments_count)):
            comment_key = df['comment_key'][i]  # comment_key = df[{your_key_column_name}][i]
            comment_body = df['comment_body'][i]  # comment_body = df[{your_text_dataset_column_name}][i]

            # if NaN, continue
            if type(comment_body) == float:
                continue

            inputs = classifier.tokenizer(comment_body,
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

            new_data = {
                'comment_key': comment_key,
                'inf_score': inf_score,
                'emt_score': emt_score
            }  # your own column names (should be unified with the columns on the 46th line)

            row_list.append(new_data)

            if i % 100 == 0:
                result_df = pd.DataFrame(row_list, columns=columns)
                if not os.path.exists(path):
                    result_df.to_csv(path, sep=',', na_rep='NaN', index=None, mode='w',
                                     header=True)
                else:
                    result_df.to_csv(path, sep=',', na_rep='NaN', index=None, mode='a', header=False)
                row_list = []

    result_df = pd.DataFrame(row_list, columns=columns)
    result_df.to_csv(path, sep=',', na_rep='NaN', index=None, mode='a', header=False)


if __name__ == "__main__":
    predict()
