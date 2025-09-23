import glob
import math

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset


def save_split_data(
    train_data: pd.DataFrame,
    max_records_per_file: int = 250000,
    train: bool = True,
    dataset_name: str = "gooaq_qa",
):

    num_files = math.ceil(len(train_data) / max_records_per_file)

    print(f"Total records: {len(train_data)}")
    print(f"Max records per file: {max_records_per_file}")
    print(f"Number of files to be created: {num_files}")

    for i in range(num_files):
        start_index = i * max_records_per_file
        end_index = start_index + max_records_per_file

        chunk_train_data = train_data.iloc[start_index:end_index]
        if train:
            output_filename = f"data/{dataset_name}_part_{i+1}_train.parquet"
        else:
            output_filename = f"data/{dataset_name}_part_{i+1}_test.parquet"
        chunk_train_data.to_parquet(output_filename, index=False)

        print(
            f"✅ Successfully saved {len(chunk_train_data)} records to {output_filename}"
        )


def recreate_dataset_from_parquets(path_pattern: str) -> Dataset:
    """Reads multiple Parquet files and creates a datasets.Dataset object."""
    # Find all files matching the pattern
    all_files = sorted(glob.glob(path_pattern))
    if not all_files:
        raise ValueError(f"No files found for pattern: {path_pattern}")

    print(f"\nFound {len(all_files)} files for pattern: '{path_pattern}'")

    # Read all Parquet files into a single pandas DataFrame
    df = pd.concat((pd.read_parquet(f) for f in all_files), ignore_index=True)

    # Convert the pandas DataFrame to a datasets.Dataset
    return Dataset.from_pandas(df)


def process_dataset(dataset_name: str):
    ds = load_dataset(dataset_name, cache_dir="data")

    if "test" not in ds and "dev" not in ds:
        print("⚠️ Test split not found. Creating a new 80/20 split.")
        # Split the training data into training and testing sets (80% train, 20% test)
        ds_split = ds["train"].train_test_split(test_size=0.2, seed=42)
        ds = DatasetDict({"train": ds_split["train"], "test": ds_split["test"]})
    # The 'gooaq' dataset uses 'dev' for its test set, so we rename it for consistency
    elif "dev" in ds:
        ds = DatasetDict({"train": ds["train"], "test": ds["dev"]})

    print("\nDataset splits configured:")
    print(ds)

    # --- Step 2: Create the partitioned files ---
    train_df = pd.DataFrame(ds["train"])
    test_df = pd.DataFrame(ds["test"])

    save_split_data(train_df, dataset_name=dataset_name, train=True)
    save_split_data(test_df, dataset_name=dataset_name, train=False)


if __name__ == "__main__":
    process_dataset(dataset_name="sentence-transformers/gooaq")
    process_dataset(dataset_name="sentence-transformers/natural-questions")

    # --- Step 3: Recreate the dataset object from the files ---
    # train_path_pattern = 'data/sentence-transformers/gooaq*_train.parquet'
    # test_path_pattern = 'data/sentence-transformers/gooaq*_test.parquet'

    # recreated_train_ds = recreate_dataset_from_parquets(train_path_pattern)
    # recreated_test_ds = recreate_dataset_from_parquets(test_path_pattern)

    # recreated_dataset = DatasetDict({
    #     'train': recreated_train_ds,
    #     'test': recreated_test_ds
    # })

    # print("\n✅ Dataset recreated successfully!")
    # print(recreated_dataset)
