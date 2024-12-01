import os
import gzip
import json
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

def process_block(dir_path):
    """
    Process each block directory to extract and aggregate data.
    """
    files = [f for f in os.listdir(dir_path) if f.endswith('.json.gz')]
    transactions = []

    for file in files:
        file_path = os.path.join(dir_path, file)
        
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            data = json.load(f)

        transaction_df = pd.json_normalize(data)

        if 'trace.structLogs' not in transaction_df or transaction_df['trace.structLogs'].apply(len).eq(0).all():
            continue

        structLogs_df = pd.json_normalize(transaction_df["trace.structLogs"][0])[['depth', 'gasCost', 'gas', 'stack', 'op']]
        # gas của CALL, DELEGATECALL là gas tổng của opcode sau khi call 
        # gas thực của CALL, DELEGATECALL sẽ được tính bằng cách lấy gas của transaction trừ đi gas của các opcode khác
        op_gasCost_df = structLogs_df[~structLogs_df['op'].isin(['CALL', 'DELEGATECALL', 'CREATE', 'CALLCODE', 'STATICCALL', 'CREATE2'])]
        
        op_gasCost = op_gasCost_df.groupby('op').agg(frequency=('op', 'count'), totalGas=('gasCost', 'sum'))
        
        # gas thực của CALL, DELEGATECALL sẽ được tính bằng cách lấy gas của transaction trừ đi gas của các opcode khác
        call_create_gas = transaction_df['trace.gas'][0] - op_gasCost['totalGas'].sum()
        call_create_freq = len(structLogs_df[structLogs_df['op'].isin(['CALL', 'DELEGATECALL', 'CREATE', 'CALLCODE', 'STATICCALL', 'CREATE2'])])
        
        op_gasCost.loc['CALL/CREATE'] = [call_create_freq, call_create_gas]
        
        transactions.append(op_gasCost)

    if transactions:
        # Extract dir_path to get folder_name only
        folder_name = os.path.basename(dir_path)
        transactions_df = pd.concat(transactions, axis=1).fillna(0)
        transactions_df[f'frequency_{folder_name}'] = transactions_df.filter(like='frequency').sum(axis=1)
        transactions_df[f'totalGas_{folder_name}'] = transactions_df.filter(like='totalGas').sum(axis=1)
        
        block_df = transactions_df[[f'frequency_{folder_name}', f'totalGas_{folder_name}']].copy()
        # dataframe chứa frequency của opcode trong cả block 
        return block_df

def main():
    path = "./../TracingData4"
    dir_paths = [os.path.join(path, d) for d in os.listdir(path) if d != '.vscode']
    dir_paths.sort(key=lambda x: int(os.path.basename(x)))

    blocks = []

    # Use ProcessPoolExecutor to parallelize the processing
    with ProcessPoolExecutor() as executor:
        for block_df in executor.map(process_block, dir_paths):
            if block_df is not None:
                blocks.append(block_df)

    blocks_df = pd.concat(blocks, axis=1).fillna(0)
    # save blocks_df to csv
    blocks_df.to_csv('./Results/OpcodeFrequency_2.csv')

if __name__ == "__main__":
    main()