#!/usr/bin/env python3
import os
import pandas as pd
from datasets import load_dataset

def row_to_dict(row, split_name):
    return {
        "image_uid": row["id"],
        "age": int(row["metadata"]["age"]),
        "sex": 1 if row["metadata"]["sex"].lower() == "male" else 2,
        "image_path": os.path.abspath(row["nii_filepath"]),
        "split": split_name
    }

def main():
    # Load the datasets
    ds_train = load_dataset("radiata-ai/brain-structure", split="train", trust_remote_code=True)
    ds_val = load_dataset("radiata-ai/brain-structure", split="validation", trust_remote_code=True)
    ds_test = load_dataset("radiata-ai/brain-structure", split="test", trust_remote_code=True)

    rows = []

    # Process each split
    for data_row in ds_train:
        rows.append(row_to_dict(data_row, "train"))
    for data_row in ds_val:
        rows.append(row_to_dict(data_row, "validation"))
    for data_row in ds_test:
        rows.append(row_to_dict(data_row, "test"))

    # Create a DataFrame and write it to CSV
    df = pd.DataFrame(rows)
    output_csv = "inputs.csv"
    df.to_csv(output_csv, index=False)
    print(f"CSV file created: {output_csv}")

if __name__ == "__main__":
    main()

