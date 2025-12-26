# Alternative approach: Download OpenWebText from the Parquet files directly
import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

# number of workers in .map() call
num_proc = 8
num_proc_load_dataset = num_proc

enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    # Try loading from monology/pile-uncopyrighted which includes OpenWebText
    # Or use the streaming approach to avoid loading scripts

    print("Attempting to load OpenWebText dataset...")

    # Try multiple approaches
    dataset = None

    # Approach 1: Try the new parquet-based version
    try:
        print("Trying: openwebtext via data_files...")
        dataset = load_dataset(
            "Skylion007/openwebtext",
            data_files="*.parquet",
            num_proc=num_proc_load_dataset
        )
        print("Success!")
    except Exception as e:
        print(f"Failed: {e}")

    # Approach 2: Try streaming and then convert
    if dataset is None:
        try:
            print("Trying: streaming approach...")
            dataset = load_dataset(
                "Skylion007/openwebtext",
                split="train",
                streaming=False
            )
            # Convert to DatasetDict
            from datasets import DatasetDict
            dataset = DatasetDict({"train": dataset})
            print("Success!")
        except Exception as e:
            print(f"Failed: {e}")

    # Approach 3: Try different revision
    if dataset is None:
        try:
            print("Trying: revision='main'...")
            dataset = load_dataset(
                "Skylion007/openwebtext",
                revision="main",
                num_proc=num_proc_load_dataset
            )
            print("Success!")
        except Exception as e:
            print(f"Failed: {e}")

    if dataset is None:
        print("ERROR: Could not load dataset with any method")
        exit(1)

    # owt by default only contains the 'train' split, so create a test split
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test') # rename the test split to val

    # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
    def process(example):
        ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
        out = {'ids': ids, 'len': len(ids)}
        return out

    # tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        # Save to scratch2 to avoid filling home directory quota
        output_dir = '/net/scratch2/junyuren/nanoGPT-manifold/data/openwebtext'
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f'{split}.bin')
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

    # train.bin is ~17GB, val.bin ~8.5MB
    # train has ~9B tokens (9,035,582,198)
    # val has ~4M tokens (4,434,897)
