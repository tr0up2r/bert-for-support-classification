import pandas as pd
import classifier

df = pd.read_csv('data/data_for_coding_inf.csv', encoding='ISO-8859-1')  # the path of your training dataset file

dataset = 'inf'  # your own column name
inf_count = df[dataset].value_counts()

possible_labels = df.inf.unique()  # your own column name
possible_labels = possible_labels[::-1]

# conduct label column
# {1: 0, 2: 1, 3: 2}
label_dict = {}
for index, possible_labels in enumerate(possible_labels):
    label_dict[possible_labels] = index
label_dict = {1: 0, 2: 1, 3: 2}

df['label'] = df.inf.replace(label_dict)

path = f'{dataset}_finetuned_BERT_epoch_'  # it will be concatenated with f'{epoch}.model'

if __name__ == "__main__":
    dataset_train, dataset_val = classifier.preparing_data(df, label_dict)
    classifier.train(dataset_train, dataset_val, label_dict, dataset, path)
