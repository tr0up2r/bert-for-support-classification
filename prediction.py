import numpy as np
import torch
from transformers import BertForSequenceClassification

import emt_classifier
import inf_classifier


inf_model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                          num_labels=len(inf_classifier.label_dict),
                                                          output_attentions=False,
                                                          output_hidden_states=False)
inf_model.to(inf_classifier.device)
inf_model.load_state_dict(torch.load('data_volume/inf_finetuned_BERT_epoch_7.model',
                                     map_location=torch.device('cpu')))

emt_model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                          num_labels=len(emt_classifier.label_dict),
                                                          output_attentions=False,
                                                          output_hidden_states=False)
emt_model.to(emt_classifier.device)
emt_model.load_state_dict(torch.load('data_volume/emt_finetuned_BERT_epoch_8.model',
                                             map_location=torch.device('cpu')))

inf_model.eval()
emt_model.eval()


# sample = emt_classifier.df.sample(n=1, random_state=1003)
# texts = sample['comment'].to_list()
texts = "test comment"
inputs = emt_classifier.tokenizer(texts,
                                  padding='max_length',
                                  return_tensors='pt')

with torch.no_grad():
    inf_outputs = inf_model(**inputs)
    emt_outputs = emt_model(**inputs)

    print(inf_outputs)
    print(emt_outputs)

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

    print(f'test_comment = {texts}')
    print(f'inf_pred_score = {inf_score}')
    print(f'emt_pred_score = {emt_score}')