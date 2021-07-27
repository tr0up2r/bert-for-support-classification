import pandas as pd

# comment data만 추출한 csv 파일을 만들기 위한 코드.
df = pd.read_csv("data/comments.csv",
                 delimiter='\t',
                 names=['comment_key', '2', '3', '4', '5', 'comment_body', '7', '8', '9', '10', '11', '12', '13'])
print(df)
print(type(df))
comments = df[['comment_key', 'comment_body']]
df_len = len(df)
deleted = 0
for i in range(df_len):
    if df['comment_body'][i] == '[removed]' or df['comment_body'][i] == '[deleted]':
        deleted += 1
print(comments)
print(deleted)

# comments.to_csv('data/only_comments.csv', sep=',', na_rep='NaN')