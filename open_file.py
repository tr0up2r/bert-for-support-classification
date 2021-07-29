import pandas as pd

# comment data만 추출한 csv 파일을 만들기 위한 코드.
df = pd.read_csv("data/comments.csv",
                 delimiter='\t',
                 names=['comment_key', '2', '3', '4', '5', 'comment_body', '7', '8', '9', '10', '11', '12', '13'])
print(df)
print(type(df))
comments = df[['comment_key', 'comment_body']]
df_len = len(df)

# deleted, removed comments mask 하기.
mask = comments['comment_body'].isin(['[deleted]', '[removed]'])
comments = comments[~mask]

print(comments)

comments.to_csv('data/only_comments.csv', sep=',', na_rep='NaN')