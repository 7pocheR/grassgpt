# OpenWebText preparation with 2024/2025 compatibility fixes
import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

# number of workers in .map() call
num_proc = 16  # Increased for faster tokenization
num_proc_load_dataset = num_proc

enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    print("Attempting to load OpenWebText dataset...")

    dataset = None

    # Approach 1: Try the parquet branch (recommended by dataset owner)
    try:
        print("\nTrying: Parquet branch (convert-parquet-full)...")
        dataset = load_dataset(
            "Skylion007/openwebtext",
            revision="convert-parquet-full",
            num_proc=num_proc_load_dataset
        )
        print("✓ Success with parquet branch!")
    except Exception as e:
        print(f"✗ Failed: {e}")

    # Approach 2: Try the alternative mirror
    if dataset is None:
        try:
            print("\nTrying: Alternative mirror (vietgpt/openwebtext_en)...")
            dataset = load_dataset(
                "vietgpt/openwebtext_en",
                num_proc=num_proc_load_dataset
            )
            print("✓ Success with alternative mirror!")
        except Exception as e:
            print(f"✗ Failed: {e}")

    if dataset is None:
        print("\n✗ ERROR: Could not load dataset with any method")
        print("Please try downgrading datasets library: pip install datasets==2.13.0")
        exit(1)

    print(f"\nDataset loaded: {dataset}")

    # owt by default only contains the 'train' split, so create a test split
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')

    print(f"Split dataset: {split_dataset}")

    # tokenize the dataset
    def process(example):
        ids = enc.encode_ordinary(example['text'])
        ids.append(enc.eot_token)
        out = {'ids': ids, 'len': len(ids)}
        return out

    print("\nTokenizing...")
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # Save to scratch2 to avoid filling home directory quota
    output_dir = '/net/scratch2/junyuren/nanoGPT-manifold/data/openwebtext'
    os.makedirs(output_dir, exist_ok=True)

    # concatenate all the ids in each dataset into one large file
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(output_dir, f'{split}.bin')
        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        print(f"\nWriting {split}.bin ({arr_len} tokens, ~{arr_len*2/1e9:.1f}GB)...")
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

        print(f"✓ Saved {filename}")

    print("\n✓ Dataset preparation complete!")
    print(f"Files saved to: {output_dir}")
