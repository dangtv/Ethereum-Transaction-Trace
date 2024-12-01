# Execution traces of Ethereum transactions

This repository contains

- A dataset of detailed execution traces of Ethereum transactions. 
- Source code for processing and analyzing the traces

## Dataset

### How to collect

To build a dataset containing detailed execution traces of Ethereum transactions, we use **Geth** client to download 3000 blocks, from four distinct periods, covering blocks from  number 19519860 to number 20050000.  We then use Geth to replay the downloaded transactions (nearly 0.7M transactions), calling `bug_traceTransaction` function to collect the execution trace.

Install **Geth** to retrieve data for an archive node: ~19TB.  

Use the command `geth sync ...` to sync and retrieve archive node data.  

1. **Collect Data**: Use `TraceCollection.py`.  

### Collected dataset

The full collected dataset can be downloaded [here](https://husteduvn-my.sharepoint.com/:f:/g/personal/dang_tranvan1_hust_edu_vn/EpXK4JEeGLRAlx-Ezpt1UFsBe49zYJwfAgiM8UGK0m4v3w?e=X9nwea)


## Data analysis

### Processing and Analysis

- Check completeness and size 
  - Use `findMissing.py` to check for any missing transactions.  
  - Use `Checksize.py` to verify if any transactions exceed size limits.  
- Use `CalculateFrequency.py` to calculate opcode frequencies. Results are stored in folder [Result](Result/)
- Recalculate the new gas usage if the caching mechanism is used via `CalculateCacheResult.py`. Results can be downloaded [here](https://drive.google.com/drive/folders/1nBPCxycWEKQOHxa4BD-DFploaVcySra9) 

### Process the results

- Running `GroupResultEvaluation.py`.  
