import os
import pandas as pd

data_types = ['4mC', '5hmC', '6mA']
for data_type in data_types:
    data_names = []
    for dir in os.listdir('data'):
        if dir.startswith(data_type):
            data_names.append(dir)
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()

    for data_name in data_names:
        now_train_data = pd.read_csv(f'data/{data_name}/train.tsv', sep='\t')
        train_data = pd.concat((train_data, now_train_data))
    train_data.drop(columns=['index'], inplace=True)
    
    for data_name in data_names:
        now_test_data = pd.read_csv(f'data/{data_name}/test.tsv', sep='\t')
        test_data = pd.concat((test_data,  now_test_data))
    test_data.drop(columns=['index'], inplace=True)

    os.makedirs(f'combined_data/{data_type}', exist_ok=True)
    train_data.to_csv(f'combined_data/{data_type}/train.tsv', sep='\t', index=False)
    test_data.to_csv(f'combined_data/{data_type}/test.tsv', sep='\t', index=False)
