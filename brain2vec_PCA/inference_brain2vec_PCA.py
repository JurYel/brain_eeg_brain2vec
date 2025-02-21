#!/usr/bin/env python3

"""
inference_brain2vec_PCA.py

Loads a pre-trained PCA-based Brain2Vec model (saved with joblib) and performs
inference on one or more input images. Produces embeddings (and optional
reconstructions) for each image.

Example usage:

    python inference_brain2vec_PCA.py \
        --pca_model pca_model.joblib \
        --input_images /path/to/img1.nii.gz /path/to/img2.nii.gz \
        --output_dir pca_output \
        --embeddings_filename pca_embeddings_2 \
        --save_recons
    
Or, if you have a CSV with image paths:

    python inference_brain2vec_PCA.py \
        --pca_model pca_model.joblib \
        --csv_input inputs.csv \
        --output_dir pca_output \
        --embeddings_filename pca_embeddings_all
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from joblib import load
import pandas as pd

from monai.transforms import (
    Compose,
    CopyItemsD,
    LoadImageD,
    EnsureChannelFirstD,
    SpacingD,
    ResizeWithPadOrCropD,
    ScaleIntensityD,
)

# Global constants
RESOLUTION = 2
INPUT_SHAPE_AE = (80, 96, 80)
FLATTENED_DIM = INPUT_SHAPE_AE[0] * INPUT_SHAPE_AE[1] * INPUT_SHAPE_AE[2]

# Reusable MONAI pipeline for preprocessing
transforms_fn = Compose([
    CopyItemsD(keys={'image_path'}, names=['image']),
    LoadImageD(image_only=True, keys=['image']),
    EnsureChannelFirstD(keys=['image']),
    SpacingD(pixdim=RESOLUTION, keys=['image']),
    ResizeWithPadOrCropD(spatial_size=INPUT_SHAPE_AE, mode='minimum', keys=['image']),
    ScaleIntensityD(minv=0, maxv=1, keys=['image']),
])


def preprocess_mri(image_path: str) -> torch.Tensor:
    """
    Preprocess an MRI using MONAI transforms to produce
    a 5D Torch tensor: (batch=1, channel=1, D, H, W).

    Args:
        image_path (str): Path to the MRI (e.g., .nii.gz file).

    Returns:
        torch.Tensor: Preprocessed 5D tensor of shape (1, 1, D, H, W).
    """
    data_dict = {"image_path": image_path}
    output_dict = transforms_fn(data_dict)
    # shape => (1, D, H, W)
    image_tensor = output_dict["image"].unsqueeze(0)  # => (1, 1, D, H, W)
    return image_tensor.float()


class PCABrain2vec(nn.Module):
    """
    A PCA-based 'autoencoder' that mimics a typical VAE interface:
      - from_pretrained(...) to load a PCA model from disk
      - forward(...) returns (reconstruction, embedding, None)

    Steps:
      1. Flatten the input volume (N, 1, D, H, W) => (N, 614400).
      2. Transform -> embeddings => shape (N, n_components).
      3. Inverse transform -> recon => shape (N, 614400).
      4. Reshape => (N, 1, D, H, W).
    """

    def __init__(self, pca_model=None):
        super().__init__()
        self.pca_model = pca_model

    def forward(self, x: torch.Tensor):
        """
        Perform a forward pass of the PCA-based "autoencoder".

        Args:
            x (torch.Tensor): Input of shape (N, 1, D, H, W).

        Returns:
            tuple(torch.Tensor, torch.Tensor, None):
                - reconstruction: (N, 1, D, H, W)
                - embedding: (N, n_components)
                - None (to align with the typical VAE interface).
        """
        n_samples = x.shape[0]
        x_cpu = x.detach().cpu().numpy()  # (N, 1, D, H, W)
        x_flat = x_cpu.reshape(n_samples, -1)  # => (N, FLATTENED_DIM)

        # PCA transform => embeddings shape (N, n_components)
        embedding_np = self.pca_model.transform(x_flat)

        # PCA inverse_transform => recon shape (N, FLATTENED_DIM)
        recon_np = self.pca_model.inverse_transform(embedding_np)
        recon_np = recon_np.reshape(n_samples, 1, *INPUT_SHAPE_AE)

        # Convert back to torch
        reconstruction_torch = torch.from_numpy(recon_np).float()
        embedding_torch = torch.from_numpy(embedding_np).float()
        return reconstruction_torch, embedding_torch, None

    @staticmethod
    def from_pretrained(pca_path: str) -> "PCABrain2vec":
        """
        Load a pre-trained PCA model (pickled or joblib) from disk.

        Args:
            pca_path (str): File path to the PCA model.

        Returns:
            PCABrain2vec: An instance wrapping the loaded PCA model.
        """
        if not os.path.exists(pca_path):
            raise FileNotFoundError(f"Could not find PCA model at {pca_path}")

        pca_model = load(pca_path)
        return PCABrain2vec(pca_model=pca_model)


def main() -> None:
    """
    Main function to parse command-line arguments and run inference
    with a pre-trained PCA Brain2Vec model.
    """
    parser = argparse.ArgumentParser(
        description="PCA-based Brain2Vec Inference Script"
    )
    parser.add_argument(
        "--pca_model", type=str, required=True,
        help="Path to the saved PCA model (.joblib)."
    )
    parser.add_argument(
        "--output_dir", type=str, default="./pca_inference_outputs",
        help="Directory to save embeddings/reconstructions."
    )
    # Two ways to supply images: multiple files or a CSV
    parser.add_argument(
        "--input_images", type=str, nargs="*",
        help="One or more image paths for inference."
    )
    parser.add_argument(
        "--csv_input", type=str, default=None,
        help="Path to a CSV containing column 'image_path'."
    )
    parser.add_argument(
        "--embeddings_filename", 
        type=str, 
        required=True,
        help="Filename (without path) to save the stacked embeddings (e.g., 'pca_embeddings.npy')."
    )
    parser.add_argument(
        "--save_recons",
        action="store_true",
        help="If set, save each reconstruction as .npy. Default is not to save."
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Build the PCA model
    pca_brain2vec = PCABrain2vec.from_pretrained(args.pca_model)
    pca_brain2vec.eval()

    # Gather image paths
    if args.csv_input:
        df = pd.read_csv(args.csv_input)
        if "image_path" not in df.columns:
            raise ValueError("CSV must contain a column named 'image_path'.")
        image_paths = df["image_path"].tolist()
    else:
        if not args.input_images:
            raise ValueError(
                "Must provide either --csv_input or --input_images."
            )
        image_paths = args.input_images

    # Inference loop
    all_embeddings = []
    for i, img_path in enumerate(image_paths):
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Preprocess
        img_tensor = preprocess_mri(img_path)

        # Forward pass
        with torch.no_grad():
            recon, embedding, _ = pca_brain2vec(img_tensor)

        # Convert to CPU numpy
        embedding_np = embedding.detach().cpu().numpy()
        recon_np = recon.detach().cpu().numpy()

        # Save (one embedding row per image)
        all_embeddings.append(embedding_np)

        # Optionally save or visualize reconstructions
        if args.save_recons:
            out_recon_path = os.path.join(args.output_dir, f"reconstruction_{i}.npy")
            np.save(out_recon_path, recon_np)
            print(f"[INFO] Saved reconstruction to: {out_recon_path}")

    # Save all embeddings stacked
    stacked_embeddings = np.vstack(all_embeddings)  # (N, n_components)
    filename = args.embeddings_filename
    if not filename.lower().endswith(".npy"):
        filename += ".npy"

    out_embed_path = os.path.join(args.output_dir, filename)
    np.save(out_embed_path, stacked_embeddings)
    print(f"[INFO] Saved embeddings of shape {stacked_embeddings.shape} to: {out_embed_path}")


if __name__ == "__main__":
    main()