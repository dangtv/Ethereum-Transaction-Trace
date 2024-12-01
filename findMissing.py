import requests
import json
import gzip
from threading import Thread, Semaphore
from pathlib import Path
import pandas as pd
import os
import csv

local_url = 'http://localhost:8545'
semaphore = Semaphore(10)  # Limit the number of concurrent threads

def generate_json_rpc(method, params, request_id=1):
    return {
        'jsonrpc': '2.0',
        'method': method,
        'params': params,
        'id': request_id,
    }

def fetch_data(method, params, url):
    rpc_data = generate_json_rpc(method, params)
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, headers=headers, json=rpc_data)
    response.raise_for_status()

    data = response.json()
    if 'error' in data:
        raise Exception(data['error']['message'])
    return data['result']

def handle_block(block_number):
    semaphore.acquire()
    try:
        block_data = fetch_data('eth_getBlockByNumber', [hex(block_number), True], local_url)
        if block_data and 'transactions' in block_data:
            transaction_hashes = [tx['hash'] for tx in block_data['transactions']]
            block_dir = Path(str(block_number))
            block_dir.mkdir(exist_ok=True)

            missing_data = []
            for index, tx_hash in enumerate(transaction_hashes):
                file_path = block_dir / f'{block_number}-{index}.json.gz'
                if not file_path.exists():
                    missing_data.append({'file_path': str(file_path), 'tx_hash': tx_hash})

            if missing_data:
                # Append missing data to CSV
                csv_file = 'missing_data.csv'
                file_exists = os.path.exists(csv_file)

                with open(csv_file, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['file_path', 'tx_hash'])
                    if not file_exists or os.path.getsize(csv_file) == 0:
                        writer.writeheader()
                    writer.writerows(missing_data)
    except Exception as e:
        print(f'Error handling block {block_number}: {e}')
    finally:
        semaphore.release()

def main():
    start_block = 19719860
    end_block = 19719860 + 1000

    threads = []

    for block_number in range(start_block, end_block):
        thread = Thread(target=handle_block, args=(block_number,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

if __name__ == '__main__':
    main()



