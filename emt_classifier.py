import pandas as pd
import classifier

df = pd.read_csv('data/data_for_coding_emt.csv', encoding='ISO-8859-1')

dataset = 'emt'
emt_count = df[dataset].value_counts()

possible_labels = df.emt.unique()
possible_labels = possible_labels[::-1]

# conduct label column
# {3: 0, 2: 1, 1: 2}
label_dict = {}
for index, possible_labels in enumerate(possible_labels):
    label_dict[possible_labels] = index
label_dict = {3: 0, 2: 1, 1: 2}

df['label'] = df.emt.replace(label_dict)

path = f'{dataset}_finetuned_BERT_epoch_'  # it will be concatenated with f'{epoch}.model'

if __name__ == "__main__":
    dataset_train, dataset_val = classifier.preparing_data(df, label_dict)
    classifier.train(dataset_train, dataset_val, label_dict, dataset, path)
