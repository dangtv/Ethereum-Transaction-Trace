import os
import gzip
import json
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import os
from multiprocessing import Pool, cpu_count
import re

def process_sstore(df):
    """
    Input: a dataframe for a group (contract, storageKey)
    Output: dataframe with updated SSTORE's newGasCost

    If last SSTORE store a value same as the value when transaction start to execute: All SSTORE is cost 100 gas (no actual SSTORE happen, just in cache)
    If not, keep gas of SSTORE as normal.
    """

    # Define a function to process each group
    def process_group_ss(gr):
        if not gr['op'].str.contains('SSTORE').any():
            return gr

        first_row = gr.query("op == 'SLOAD' or op == 'SSTORE'").iloc[0]
        last_sstore = gr[gr['op'] == 'SSTORE'].iloc[-1]

        if first_row['op'] == 'SLOAD':
            if first_row['storageValue'] != last_sstore['storageValue']:
                return gr
            else:
                gr.loc[gr['op'] == 'SSTORE', 'newGasCost'] = 100
                return gr

        return gr

    # Apply the processing function to each group and concatenate the results
    df = df.groupby('transaction').apply(process_group_ss, include_groups=False)
    
    return df

def process_sstore_2(df): 
    """ 
    Input: a dataframe for sstore of a contract's storageValue in block 
    Output: dataframe with updated SSTORE's newGasCost 
 
    Case 1: First row is SLOAD and has the same value as final SSTORE: All is warm (100 gas) 
    Case 2: If not case 1: then all but last SSTORE in df is warm access (100 gas) 
    """ 
 
    sload_sstore_df = df.query("op == 'SLOAD' or op == 'SSTORE'")
    if not sload_sstore_df.empty: 
        first_row = sload_sstore_df.iloc[0] 
        sstore_df = df[df['op'] == 'SSTORE']
        if not sstore_df.empty:
            last_sstore = sstore_df.iloc[-1] 
            if first_row['op'] == 'SLOAD' and first_row['storageValue'] == last_sstore['storageValue']: 
                # Case 1: All SSTORE operations are warm
                df.loc[df['op'] == 'SSTORE', 'newGasCost'] = 100 
            else: 
                # Case 2: All SSTORE operations except the last are warm
                sstore_indices = sstore_df.index 
                if not sstore_indices.empty: 
                    first_sstore_gas = df.loc[sstore_indices[0], 'gasCost']
                    df.loc[sstore_indices[-1], 'newGasCost'] = first_sstore_gas 
                    df.loc[sstore_indices[:-1], 'newGasCost'] = 100
    return df


def process_sload(df):
    """
    Input: a dataframe for sload of a contract's storageValue in block
    Output: dataframe with updated SLOAD's newGasCost

    Case 1: First row is SLOAD: Except that SLOAD is cold access (2100 gas), all follow up is warm (100 gas)
    Case 2: First row is SSTORE: All SLOAD in df is warm access (100 gas)
    """
    if len(df.query("op == 'SLOAD' or op == 'SSTORE'")) > 0:
        # Filter the dataframe for the first SLOAD or SSTORE operation
        first_op = df.query("op == 'SLOAD' or op == 'SSTORE'").iloc[0]
        
        if first_op['op'] == 'SSTORE':
            # Case 2: First row is SSTORE, all SLOAD are warm access (100 gas)
            df.loc[df['op'] == 'SLOAD', 'newGasCost'] = 100
        else:
            # Case 1: First row is SLOAD, update newGasCost accordingly
            sload_indices = df[df['op'] == 'SLOAD'].index
            if not sload_indices.empty:
                # Set the first SLOAD's newGasCost to its original gasCost
                df.loc[sload_indices[0], 'newGasCost'] = df.loc[sload_indices[0], 'gasCost']
                # Set the rest of the SLOAD's newGasCost to 100
                df.loc[sload_indices[1:], 'newGasCost'] = 100
    
    return df


class TransactionDataProcessor:
    def __init__(self, directory_path):
        self.directory_path = directory_path

    def load_data(self, file_path):
        with gzip.open(file_path, 'rt', encoding='utf-8') as file:
            data = json.load(file)
        return data

    def process_data(self, data, file_name):
        df = pd.json_normalize(data)

        # Check if 'transaction.accessList is emtpy or not present
        if 'transaction.accessList' in df and df['transaction.accessList'][0]:
            OAL = pd.json_normalize(df['transaction.accessList'][0])
            OAL1 = OAL.explode('storageKeys')
            OAL1 = OAL1.dropna(subset=['storageKeys'])

            # Reset the index
            OAL1.reset_index(drop=True, inplace=True)

            # Add new columns
            OAL1['gasCost'] = 1900
            OAL1['output'] = None
            OAL1['op'] = 'OAL'
            OAL1['index'] = OAL1.index
            OAL1['transaction'] = file_name.split(".json.gz")[0]

            # Rename columns
            OAL1 = OAL1.rename(columns={'storageKeys': 'params', 'address' : 'cur_contract'})

        else:
            OAL1 = pd.DataFrame(columns=['gasCost', 'params', 'op', 'output', 'cur_contract', 'index'])

        OAL1['cur_contract_code'] = None


        # Check if 'trace.structLogs' is empty or not present
        if 'trace.structLogs' not in df or df['trace.structLogs'].apply(len).eq(0).all():
            return None, None

        df1 = pd.json_normalize(df['trace.structLogs'][0]).iloc[:, :7]
        df1['transaction'] = file_name.split(".json.gz")[0] 

        def contract_interaction(df, initial_contract_address):
            # Initialize the new columns
            df['cur_contract_code'] = None
            df['cur_contract'] = None

            # Stack to keep track of contract addresses at each depth for code and state
            contract_codestate_stack = [initial_contract_address]
            contract_state_stack = [initial_contract_address]
            
            # Iterate over the DataFrame rows
            for index, row in df.iterrows():
                # If there's a call opcode, extract the 'to' address from the stack
                if row['op'] in ['CALL', 'CALLCODE', 'DELEGATECALL', 'STATICCALL', 'CREATE', 'CREATE2']:
                    stack = row['stack']
                
                    if row['op'] in ['CALL', 'STATICCALL']:
                        new_contract_address = stack[-2]
                        # Both code and state are from the called contract
                        contract_codestate_stack.append(new_contract_address)
                        contract_state_stack.append(new_contract_address)

                    elif row['op'] in ['CREATE', 'CREATE2']:
                        # find the index of next nearest row have the same 'depth' value as current one
                        next_index = index + 1
                        while next_index < len(df) and df.at[next_index, 'depth'] != row['depth']:
                            next_index += 1

                        new_contract_address = df.at[next_index, 'stack'][-1]
                        contract_codestate_stack.append(new_contract_address)
                        contract_state_stack.append(new_contract_address)

                    elif row['op'] in ['DELEGATECALL', 'CALLCODE']:
                        new_contract_address = stack[-2]
                        # Code is from the called contract, state is from the current contract
                        contract_codestate_stack.append(new_contract_address)
                        contract_state_stack.append(contract_state_stack[-1])

                # If the depth decreases, pop the last contract address from the stacks
                elif len(contract_codestate_stack) > row['depth']:
                    contract_codestate_stack.pop()
                    contract_state_stack.pop()
                
                # First row
                if index == 0:
                    df.at[index, 'cur_contract_code'] = contract_codestate_stack[-1]
                    df.at[index, 'cur_contract'] = contract_state_stack[-1]
                
                # Index + 1 because the opcode execute is in context of previous contract
                if index != len(df1)-1:
                    df.at[index + 1, 'cur_contract_code'] = contract_codestate_stack[-1]
                    df.at[index + 1, 'cur_contract'] = contract_state_stack[-1]

            return df

        # Contract Interaction Trace Analysis
        initial_contract_address = df['transaction.to'][0]
        if initial_contract_address == None:
            initial_contract_address = df['receipt.contractAddress'][0]
        df1 = contract_interaction(df1, initial_contract_address)

        return df1, OAL1

    def filter_and_extract(self, df1, OAL1):
        sload_df = df1[df1['op'].isin(['SLOAD', 'SSTORE'])].copy()
        sload_df['params'] = sload_df.apply(self.extract_params, axis=1)

        if sload_df.empty:
            columns = ['transaction', 'cur_contract', 'cur_contract_code', 'gasCost', 'op', 'params', 'output']
            return pd.DataFrame(columns=columns)

        sload_df['output'] = sload_df.apply(lambda row: self.extract_output(row, df1['stack'].shift(-1)[row.name]), axis=1)
        new_df = sload_df[['transaction', 'cur_contract', 'cur_contract_code', 'gasCost', 'op', 'params', 'output']].copy()
        new_df['index'] = new_df.index
        new_df.reset_index(drop=True, inplace=True)
        new_df = new_df.sort_values(['cur_contract', 'index'])

        # Function to shorten storageKey hex values
        def shorten_hex_storageKey(hex_str):
            # Remove leading zeros and ensure at least one character remains
            # Ensure hex_str is a string
            hex_str = hex_str[2:]
            shortened = hex_str.lstrip('0')
            return '0x' + (shortened if shortened else '0')

        # Function to correct contract address
        def correct_address(address):
            # Remove '0x' prefix, add leading zeros if necessary, and re-add '0x' prefix
            if address is None:
                return None
            else:
                # Remove '0x' prefix, add leading zeros if necessary, and re-add '0x' prefix
                stripped_address = address[2:]  # Remove '0x' prefix

                # Address correction for length
                if len(stripped_address) > 40:
                    # If the address exceeds 20 bytes, trim it to the last 20 bytes (40 characters)
                    corrected_address = '0x' + stripped_address[-40:]
                else:
                    # Add leading zeros if the address is less than 20 bytes
                    corrected_address = '0x' + stripped_address.rjust(40, '0')

                return corrected_address

        # Apply functions to the DataFrame
        if not OAL1.empty:
            OAL1['params'] = OAL1['params'].apply(shorten_hex_storageKey)

        if not new_df.empty:
            new_df['cur_contract'] = new_df['cur_contract'].apply(correct_address)
            new_df['cur_contract_code'] = new_df['cur_contract_code'].apply(correct_address)

        new_df = pd.concat([OAL1, new_df])
        return new_df

    @staticmethod
    def extract_params(row):
        stack = row['stack']
        op = row['op']
        if not stack:
            return None
        if op == 'SLOAD':
            return stack[-1:][0]
        elif op == 'SSTORE':
            return stack[-1:][0]
        else:
            return None

    @staticmethod
    def extract_output(row, next_stack):
        op = row['op']
        if op in ['SLOAD'] and next_stack:
            # Return the last element of the next stack as a single value
            return next_stack[-1] if next_stack else None
        elif op == 'SSTORE':
            stack = row['stack']
            return stack[-2:-1][0]
        else:
            # For other operations, return None or appropriate single value
            return None

    def process(self, folder_name):
        all_dfs = []  # List to store all DataFrames
 
        def natural_sort_key(s):
            import re
            return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

        # Get all files in the directory
        files = [f for f in os.listdir(self.directory_path) if os.path.isfile(os.path.join(self.directory_path, f))]

        # Sort the files using natural sorting
        sorted_files = sorted(files, key=natural_sort_key)

        for filename in sorted_files:
            if filename.endswith('.json.gz'):
                file_path = os.path.join(self.directory_path, filename)
                data = self.load_data(file_path)
                df1, OAL1 = self.process_data(data, filename)
                if df1 is not None:
                    new_df = self.filter_and_extract(df1, OAL1)
                    all_dfs.append(new_df)  # Append the processed DataFrame to the list

        # Concatenate all DataFrames into one
        final_df = pd.concat(all_dfs, ignore_index=True)

        # Reset the index
        final_df = final_df.reset_index(drop=True)

        final_df['new_index'] = final_df.index

        # Sort the DataFrame based on 'cur_contract' and 'index'
        final_df = final_df.sort_values(['cur_contract', 'new_index'])

        # Rename columns
        df = final_df.rename(columns={'cur_contract' : 'contract', 'params' : 'storageKey', 'output' : 'storageValue'})

        # Initialize the newGasCost column
        df['newGasCost'] = df['gasCost']
        df['block'] = folder_name

        # Function to process each group
        def process_group(group):
            # Handle SLOAD operations
            group = process_sload(group)
            # Handle SSTORE operations
            group = process_sstore_2(group)
            
            return group

        # Group by 'contract' and 'storageKey' and apply processing
        df1 = df.groupby(['contract', 'storageKey']).apply(process_group, include_groups=False)
        df1 = df1.reset_index()
        # df1 = df1.drop("level_3", axis=1)

        return df1
    

import os
from multiprocessing import Pool, cpu_count
import re

def process_folder(folder_path):
    """
    Worker function to process each folder.
    """
    try:
        folder_name = os.path.basename(folder_path)
        match = re.search(r'\d+', folder_name)
        name = match.group()
        processor = TransactionDataProcessor(folder_path)
        df = processor.process(name)
        return df
    except Exception as e:
        print(f"An error occurred while processing '{folder_path}': {e}")
        return None

def main():
    tracing_data_paths = [r'../TracingData', r'../TracingData2', r'../TracingData3', r'../TracingData4']
    output_folder = './Result'

    for i, tracing_data_path in enumerate(tracing_data_paths):
        if not os.path.exists(tracing_data_path):
            print(f"Directory '{tracing_data_path}' does not exist.")
            return
        
        folder_names = [name for name in os.listdir(tracing_data_path) if os.path.isdir(os.path.join(tracing_data_path, name))]
        folder_names.sort()

        folder_paths = [os.path.join(tracing_data_path, name) for name in folder_names]

        final_result = []

        with Pool(processes=12) as pool:
            results = pool.map(process_folder, folder_paths)

        # Filter out None results in case of errors
        results = [res for res in results if res is not None]

        # Concat the dataframes
        final_result = pd.concat(results, ignore_index=True)

        # Optionally, save the final_result DataFrame to a file
        final_result.to_csv(os.path.join(output_folder, f'CacheResult_{i}{i}.csv'), index=False)

if __name__ == '__main__':
    main()