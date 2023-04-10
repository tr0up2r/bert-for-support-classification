# bert-for-support-classification  
BERT multi-class classifiers that assign the **IS<sub>Informational Support</sub>** and **ES<sub>Emotional Support</sub> scores** on a 3 point Likert scale to text data.  

## How to fine-tune your own multi-class support classifier
1. Run **inf_classifier.py** or **emt_classifier.py** depending on the model you want to build. Before running, please check the following:
* **Check the format of training dataset file**: In my case, the columns of training dataset file are ['comment', 'inf' (or ‘emt’)] (‘comment’: ‘str’ type of text datasets, ‘inf’ (or ‘emt’): label for that text dataset (1-3 points)).  
If the column name corresponding to text dataset is different from my case, please modify the **36th and 45th line in the classifier.py** file.  

```python
encoded_data_train = tokenizer.batch_encode_plus(
    df[df.data_type == 'train'].comment.values,  # df[df.data_type == 'train'].{your_column_name}.values
    add_special_tokens=True,
    return_attention_mask=True,
    pad_to_max_length=True,
    max_length=256,
    return_tensors='pt'
    )
    
encoded_data_val = tokenizer.batch_encode_plus(
    df[df.data_type == 'val'].comment.values,  # df[df.data_type == 'val'].{your_column_name}.values
    add_special_tokens=True,
    return_attention_mask=True,
    pad_to_max_length=True,
    max_length=256,
    return_tensors='pt'
    )
```

* **4th line of inf_classifier.py and emt_classifier.py**: Change it to the path of your dataset file.  

```python
df = pd.read_csv('data/data_for_coding_inf.csv', encoding='ISO-8859-1')  # the path of your training dataset file
```

* **6th and 9th line of inf_classifier.py and emt_classifier.py**: Change it to the label column of your file.  
```python
dataset = 'inf'  # your own column name
```
```python
possible_labels = df.inf.unique()  # your own column name
```

* **126th to 140th line of classifier.py**: You can change the hyperparameter values.  
```python
# adjust hyper parameters
batch_size = 3
dataloader_train = DataLoader(dataset_train,
                              sampler=RandomSampler(dataset_train),
                              batch_size=batch_size)
dataloader_validation = DataLoader(dataset_val,
                                   sampler=SequentialSampler(dataset_val),
                                   batch_size=batch_size)
optimizer = AdamW(model.parameters(),
                  lr=1e-5,
                  eps=1e-8)
epochs = 10
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train) * epochs)
```

2. After running, you can check the learning results in the **{dataset}_training_result.csv file**. The states of models are also saved.  

----

## How to predict
1. Run **inf_classifier.py or emt_classifier.py** to save the classifier status as **.model** file format before prediction.  

2. Run **prediction.py.** Before running prediction.py, please check the following:
* **42nd line of prediction.py**: Enter the path of the file containing the dataset you want to predict.  
In my case, the columns of dataset file are [‘comment_key’, ‘comment_body’]. You can run this file without modification by specifying the key of the document you want to predict as ‘comment_key’ and document’s contents as ‘comment_body’, respectively.  
```python
df = pd.read_csv("data/only_comments.csv")  # the path of your dataset file want to predict  
```

If you want to change the source code according to your dataset, make additional modifications to the 51 and 52nd lines.
```python
comment_key = df['comment_key'][i]  # comment_key = df[{your_key_column_name}][i]
comment_body = df['comment_body'][i]  # comment_body = df[{your_text_dataset_column_name}][i]
```

* **46th and 84th to 88th lines**: Determine the format of the file where the results will be saved.  
In my case, I specified the columns of the result file as [‘comment_key’, ‘inf_score (informational support score)’, ‘emt_score (emotional support score)’]. Also, you can change the format of the values to be stored in the 84th line. This would have to be unified with the columns mentioned above.  
```python
columns = ['comment_key', 'inf_score', 'emt_score']  # your own column names
```
```python
new_data = {
    'comment_key': comment_key,
    'inf_score': inf_score,
    'emt_score': emt_score
    }  # your own column names (should be unified with the columns on the 46th line)
```

* **47th lines**: Specifies the path where the result file will be saved.  
```python
path = 'prediction_results.csv'  # your own file path
```
