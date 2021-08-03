import pandas as pd
import csv


def inf_emt_count(result_df):
    score = [0, 0, 0, 0, 0, 0]

    df_len = len(result_df)

    for i in range (df_len):
        # str 형으로 저장되다가 int 형으로 저장된 문제가 있어, 조건문을 아래와 같이 설정.
        if result_df['inf_score'][i] == '1' or result_df['inf_score'][i] == 1:
            score[0] += 1
        elif result_df['inf_score'][i] == '2' or result_df['inf_score'][i] == 2:
            score[1] += 1
        else:
            score[2] += 1

        if result_df['emt_score'][i] == '1' or result_df['emt_score'][i] == 1:
            score[3] += 1
        elif result_df['emt_score'][i] == '2' or result_df['emt_score'][i] == 2:
            score[4] += 1
        else:
            score[5] += 1

    return score


df = pd.read_csv("data/prediction_results_without_duplicates.csv")
print(df)

result = df[['comment_key', 'inf_score', 'emt_score']]
result.to_csv('data/prediction_results_without_duplicates.csv', sep=',', na_rep='NaN', index=None)


inf_emt_score = inf_emt_count(df)

with open("data/prediction_results_count_without_duplicates.csv", "a") as fp:
    wr = csv.writer(fp, dialect='excel')
    wr.writerow(inf_emt_score)


# link_id와 parent_id가 같은 경우에 대한 처리.
idx = df[df['link_id'] == df['parent_id']].index
masked_df = df.drop(idx)

print(masked_df)

# drop 후 바로 count를 진행하면 key error 발생.
# csv로 저장 후, 다시 불러와서 count 진행하였음.
masked_df.to_csv('data/masked_prediction_results_without_duplicates.csv', sep=',', na_rep='NaN', index=None)
masked_df = pd.read_csv("data/masked_prediction_results_without_duplicates.csv")

inf_emt_score2 = inf_emt_count(masked_df)

with open("data/masked_prediction_results_count_without_duplicates.csv", "a") as fp:
    wr = csv.writer(fp, dialect='excel')
    wr.writerow(inf_emt_score2)