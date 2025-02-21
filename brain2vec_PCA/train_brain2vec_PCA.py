#!/usr/bin/env python3

"""
train_brain2vec_PCA.py

A PCA-based "autoencoder" script for brain MRI data, with support for both
incremental PCA and standard PCA. Only scans labeled 'train' in the CSV
(split == 'train') will be used for fitting.

Example usage:
    python train_brain2vec_PCA.py \
        --inputs_csv /path/to/inputs.csv \
        --output_dir ./pca_outputs \
        --pca_type standard \
        --n_components 1200
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from monai import transforms
from monai.data import Dataset, PersistentDataset
from monai.transforms.transform import Transform
from sklearn.decomposition import PCA, IncrementalPCA
from typing import Optional, Union, Tuple

# voxel resolution
RESOLUTION = 2

# cropped image dimensions after transform
INPUT_SHAPE_AE = (80, 96, 80)

DEFAULT_N_COMPONENTS = 1200


def get_dataset_from_pd(
    df: pd.DataFrame,
    transforms_fn: Transform,
    cache_dir: Optional[str]
) -> Union[Dataset, PersistentDataset]:
    """
    Create a MONAI Dataset or PersistentDataset from the given DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with at least 'image_path' column.
        transforms_fn (Transform): MONAI transform pipeline.
        cache_dir (Optional[str]): If provided, use PersistentDataset caching.

    Returns:
        Dataset|PersistentDataset: A dataset for training or inference.
    """
    data_dicts = df.to_dict(orient='records')
    if cache_dir and cache_dir.strip():
        os.makedirs(cache_dir, exist_ok=True)
        dataset = PersistentDataset(
            data=data_dicts,
            transform=transforms_fn,
            cache_dir=cache_dir
        )
    else:
        dataset = Dataset(data=data_dicts, transform=transforms_fn)
    return dataset


class PCAAutoencoder:
    """
    A PCA 'autoencoder' that can use either standard PCA or IncrementalPCA:
      - fit(X): trains the model
      - transform(X): get embeddings
      - inverse_transform(Z): reconstruct data from embeddings
      - forward(X): returns (X_recon, Z).
    
    If using standard PCA, a single call to .fit(X) is made.
    If using incremental PCA, .partial_fit is called in batches.
    """

    def __init__(
        self,
        n_components: int = DEFAULT_N_COMPONENTS,
        batch_size: int = 128,
        pca_type: str = 'standard'
    ) -> None:
        """
        Initialize the PCAAutoencoder.

        Args:
            n_components (int): Number of principal components to keep.
            batch_size (int): Chunk size for partial_fit or chunked transform.
            pca_type (str): Either 'incremental' or 'standard'.
        """
        self.n_components = n_components
        self.batch_size = batch_size
        self.pca_type = pca_type.lower()

        if self.pca_type == 'incremental':
            self.ipca = IncrementalPCA(n_components=self.n_components)
        else:
            # Default to standard PCA
            self.ipca = PCA(n_components=self.n_components, svd_solver='randomized')

    def fit(self, X: np.ndarray) -> None:
        """
        Fit the PCA model. If incremental PCA, calls partial_fit in batches;
        otherwise calls .fit once on the entire data array.

        Args:
            X (np.ndarray): Shape (n_samples, n_features).
        """
        if self.pca_type == 'standard':
            self.ipca.fit(X)
        else:
            # IncrementalPCA
            n_samples = X.shape[0]
            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                self.ipca.partial_fit(X[start_idx:end_idx])

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project data into the PCA latent space in batches for memory efficiency.

        Args:
            X (np.ndarray): Shape (n_samples, n_features).

        Returns:
            np.ndarray: Latent embeddings of shape (n_samples, n_components).
        """
        results = []
        n_samples = X.shape[0]
        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            Z_chunk = self.ipca.transform(X[start_idx:end_idx])
            results.append(Z_chunk)
        return np.vstack(results)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """
        Reconstruct data from PCA latent space in batches.

        Args:
            Z (np.ndarray): Latent embeddings of shape (n_samples, n_components).

        Returns:
            np.ndarray: Reconstructed data of shape (n_samples, n_features).
        """
        results = []
        n_samples = Z.shape[0]
        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            X_chunk = self.ipca.inverse_transform(Z[start_idx:end_idx])
            results.append(X_chunk)
        return np.vstack(results)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mimic a linear AE's forward() returning (X_recon, Z).

        Args:
            X (np.ndarray): Original data of shape (n_samples, n_features).

        Returns:
            tuple[np.ndarray, np.ndarray]: (X_recon, Z).
        """
        Z = self.transform(X)
        X_recon = self.inverse_transform(Z)
        return X_recon, Z


def load_and_flatten_dataset(
    csv_path: str,
    cache_dir: str,
    transforms_fn: Transform
) -> np.ndarray:
    """
    Load and flatten MRI volumes from the provided CSV.

    1) Reads CSV.
    2) Filters rows if 'split' in columns => only keep rows with split == 'train'.
    3) Applies transforms to each image, flattening them into a 1D vector.
    4) Returns a NumPy array X of shape (n_samples, 614400) after flattening.

    Args:
        csv_path (str): Path to a CSV containing at least 'image_path' column. 
                        Optionally has a 'split' column.
        cache_dir (str): Path to cache directory for MONAI PersistentDataset.
        transforms_fn (Transform): MONAI transform pipeline.

    Returns:
        np.ndarray: Flattened image data of shape (n_samples, 614400).
    """
    df = pd.read_csv(csv_path)

    # Keep only 'train' samples if split column exists
    if 'split' in df.columns:
        df = df[df['split'] == 'train']

    dataset = get_dataset_from_pd(df, transforms_fn, cache_dir)
    loader = DataLoader(dataset, batch_size=1, num_workers=0)

    X_list = []
    for batch in loader:
        # batch["image"] => shape (1, 1, 80, 96, 80)
        img = batch["image"].squeeze(0)  # => shape (1, 80, 96, 80)
        flattened = img.numpy().flatten()  # => (614400,)
        X_list.append(flattened)

    if not X_list:
        raise ValueError(
            "No training samples found (split='train'). Check your CSV or 'split' values."
        )

    X = np.vstack(X_list)
    return X


def main() -> None:
    """
    Main function to parse command-line arguments and fit a PCA or IncrementalPCA model,
    then save embeddings and reconstructions.
    """
    parser = argparse.ArgumentParser(
        description="PCA Autoencoder with MONAI transforms and 'split' filtering."
    )
    parser.add_argument(
        "--inputs_csv", type=str, required=True,
        help="Path to CSV with at least 'image_path' column and optional 'split' column."
    )
    parser.add_argument(
        "--cache_dir", type=str, default="",
        help="Cache directory for MONAI PersistentDataset (optional)."
    )
    parser.add_argument(
        "--output_dir", type=str, default="./pca_outputs",
        help="Where to save PCA model and embeddings."
    )
    parser.add_argument(
        "--batch_size_ipca", type=int, default=128,
        help="Batch size for partial_fit or chunked transform."
    )
    parser.add_argument(
        "--n_components", type=int, default=1200,
        help="Number of PCA components to keep."
    )
    parser.add_argument(
        "--pca_type", type=str, default="incremental",
        choices=["incremental", "standard"],
        help="Which PCA algorithm to use: 'incremental' or 'standard'."
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    transforms_fn = transforms.Compose([
        transforms.CopyItemsD(keys={'image_path'}, names=['image']),
        transforms.LoadImageD(image_only=True, keys=['image']),
        transforms.EnsureChannelFirstD(keys=['image']),
        transforms.SpacingD(pixdim=RESOLUTION, keys=['image']),
        transforms.ResizeWithPadOrCropD(
            spatial_size=INPUT_SHAPE_AE, mode='minimum', keys=['image']
        ),
        transforms.ScaleIntensityD(minv=0, maxv=1, keys=['image']),
    ])

    print("Loading and flattening dataset from:", args.inputs_csv)
    X = load_and_flatten_dataset(args.inputs_csv, args.cache_dir, transforms_fn)
    print(f"Dataset shape after flattening: {X.shape}")

    # Build the PCAAutoencoder with chosen type
    model = PCAAutoencoder(
        n_components=args.n_components,
        batch_size=args.batch_size_ipca,
        pca_type=args.pca_type
    )

    # Fit the PCA model
    print(f"Fitting {args.pca_type.capitalize()}PCA")
    model.fit(X)
    print("Done fitting PCA. Transforming data to embeddings...")

    # Get embeddings & reconstruction
    X_recon, Z = model.forward(X)
    print("Embeddings shape:", Z.shape)
    print("Reconstruction shape:", X_recon.shape)

    # Save embeddings and reconstructions
    embeddings_path = os.path.join(args.output_dir, "pca_embeddings.npy")
    recons_path = os.path.join(args.output_dir, "pca_reconstructions.npy")
    np.save(embeddings_path, Z)
    np.save(recons_path, X_recon)
    print(f"Saved embeddings to {embeddings_path}")
    print(f"Saved reconstructions to {recons_path}")

    # Optionally save the actual PCA model with joblib
    from joblib import dump
    ipca_model_path = os.path.join(args.output_dir, "pca_model.joblib")
    dump(model.ipca, ipca_model_path)
    print(f"Saved PCA model to {ipca_model_path}")


if __name__ == "__main__":
    main()