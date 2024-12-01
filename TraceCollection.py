import requests
import json
import gzip
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import sleep
from multiprocessing import Pool, cpu_count
from threading import Semaphore
import gc
from requests.exceptions import ChunkedEncodingError
import time

provider_url = 'http://localhost:8545'
local_url = 'http://localhost:8545'
rate_limit = 200  # Maximum API calls per second for provider_url
semaphore = Semaphore(rate_limit)  # Semaphore for rate limiting

def generate_json_rpc(method, params, request_id=1):
    return {
        'jsonrpc': '2.0',
        'method': method,
        'params': params,
        'id': request_id,
    }

def fetch_data(method, params, url, max_retries=5):
    rpc_data = generate_json_rpc(method, params)
    headers = {'Content-Type': 'application/json'}
    retries = 0

    while retries < max_retries:
        try:
            with requests.post(url, headers=headers, json=rpc_data, stream=True) as response:
                response.raise_for_status()
                chunks = response.iter_content(chunk_size=8192)
                raw_data = b''.join(chunks)
                data = json.loads(raw_data.decode('utf-8'))
                if 'error' in data:
                    raise Exception(data['error']['message'])
                return data['result']
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError occurred: {e}")
            raise
        except ChunkedEncodingError as e:
            retries += 1
            sleep(1)  # Wait before retrying
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            raise

    raise Exception(f"Failed to fetch data after {max_retries} retries.")

def save_transaction_data(block_number, tx_index, transaction_data):
    if not transaction_data or 'error' in transaction_data:
        print(f"Error in transaction data for block {block_number}, index {tx_index}: {transaction_data.get('error')}")
        return
    
    folder_name = f"{block_number}"
    folder_path = Path.cwd() / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)

    file_name = f"{block_number}-{tx_index}.json.gz"
    file_path = folder_path / file_name

    json_content = json.dumps(transaction_data, indent=2)
    with gzip.open(file_path, 'wt', encoding='utf-8') as f:
        f.write(json_content)

def save_failed_transaction(tx_hash, block_number, tx_index):
    folder_name = 'failed_transactions'
    folder_path = Path.cwd() / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)

    file_name = 'failed_transactions.txt'
    file_path = folder_path / file_name

    content = f"{tx_hash}\n"

    print(f"Appending failed transaction {tx_hash} for block {block_number}, index {tx_index} to {file_name}")

    with open(file_path, 'a') as f:
        f.write(content)
        print(f"Successfully appended failed transaction {tx_hash} to {file_name}")

def transaction_file_exists(block_number, tx_index):
    file_name = f"{block_number}-{tx_index}.json.gz"
    file_path = Path.cwd() / f"{block_number}" / file_name
    return file_path.exists()

def process_transaction(tx_hash, block_number, tx_index):
    if transaction_file_exists(block_number, tx_index):
        return

    try:
        transaction = fetch_data('eth_getTransactionByHash', [tx_hash], local_url)
        receipt = fetch_data('eth_getTransactionReceipt', [tx_hash], local_url)
        # code = None
        # if transaction.get('to'):
        #     code = fetch_data('eth_getCode', [transaction['to'], 'latest'], local_url)
        trace = fetch_data_with_rate_limit('debug_traceTransaction', [tx_hash, {'timeout': '240s', 'reexec': 1005, 'disableMemory': False, 'disableStorage': True, 'disableReturnData': True}], provider_url)
        transaction_data = {'transaction': transaction, 'receipt': receipt, 'trace': trace}  # 'code': code
        save_transaction_data(block_number, tx_index, transaction_data)
        gc.collect()  # Explicitly trigger garbage collection to free up memory
    except Exception as e:
        print(f"Error processing transaction {tx_hash}: {e}")
        save_failed_transaction(tx_hash, block_number, tx_index)

def fetch_data_with_rate_limit(method, params, url):
    semaphore.acquire()
    try:
        return fetch_data(method, params, url)
    finally:
        semaphore.release()

def handle_block(block_number):
    try:
        block_data = fetch_data('eth_getBlockByNumber', [hex(block_number), True], local_url)
        if block_data and 'transactions' in block_data:
            transaction_hashes = [tx['hash'] for tx in block_data['transactions']]
            process_transactions_in_block(transaction_hashes, block_number)
        else:
            print(f"Failed to fetch block data for block number: {block_number}")
        print(f"Successfully processed block {block_number}")
    except Exception as e:
        print(f"Exception occurred while handling block {block_number}: {e}")

def process_transactions_in_block(transaction_hashes, block_number):
    for index, tx_hash in enumerate(transaction_hashes):
        process_transaction(tx_hash, block_number, index)

def main():
    # lấy block gần nhất 
    # lấy 1000 block gần nhất 
    # lấy 200000 block gần 
    # 20117791 lấy tại thời điểm tháng 
    # https://etherscan.io/block/20117791
    start_block = 20117791
    end_block = 20117792

    # num_cores = cpu_count()
    block_numbers = range(start_block, end_block)

    # mỗi process chạy mất 5GB 
    with Pool(processes=2) as pool:
        pool.map(handle_block, block_numbers)

if __name__ == "__main__":
    main()
