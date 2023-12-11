import pandas as pd

SEED = 42

# test_df = pd.read_csv('./RewardRepair/D4JPairs.csv', encoding='latin-1', delimiter='\t')
test_df = pd.read_csv('./RewardRepair/valid_data.csv', encoding='latin-1', delimiter='\t')
print(test_df.head())
test_df = test_df[['bugid', 'buggy', 'patch']]
print(test_df.head())

print(test_df.iloc[1]['bugid'])
print(test_df.iloc[1]['buggy'])
print(test_df.iloc[1]['buggy'].find('\t'))
print(test_df.iloc[1]['buggy'].find('\n'))
print(test_df.iloc[1]['patch'])

# test_dataset = test_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
# print("TEST Dataset: {}".format(test_dataset.shape))
