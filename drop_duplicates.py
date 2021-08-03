import pandas as pd

df = pd.read_csv("data/prediction_results.csv")
print(df)

dup = df.drop_duplicates()
print(dup)
dup.to_csv('data/prediction_results_without_duplicates.csv', sep=',', na_rep='NaN', index=None)