#!/usr/bin/env python3

"""
inference_brain2vec.py

Loads a pretrained Brain2vec VAE (AutoencoderKL) model and performs inference
on one or more MRI images, generating reconstructions and latent parameters 
(z_mu, z_sigma).

Example usage:

    # 1) Multiple file paths
    python inference_brain2vec.py \
        --checkpoint_path /path/to/autoencoder_checkpoint.pth \
        --input_images /path/to/img1.nii.gz /path/to/img2.nii.gz \
        --output_dir ./vae_inference_outputs \
        --device cuda

    # 2) Use a CSV containing image paths
    python inference_brain2vec.py \
        --checkpoint_path /path/to/autoencoder_checkpoint.pth \
        --csv_input /path/to/images.csv \
        --output_dir ./vae_inference_outputs
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from typing import Optional
from monai.transforms import (
    Compose,
    CopyItemsD,
    LoadImageD,
    EnsureChannelFirstD,
    SpacingD,
    ResizeWithPadOrCropD,
    ScaleIntensityD,
)
from generative.networks.nets import AutoencoderKL
import pandas as pd


RESOLUTION = 2
INPUT_SHAPE_AE = (80, 96, 80)

transforms_fn = Compose([
    CopyItemsD(keys={'image_path'}, names=['image']),
    LoadImageD(image_only=True, keys=['image']),
    EnsureChannelFirstD(keys=['image']),
    SpacingD(pixdim=RESOLUTION, keys=['image']),
    ResizeWithPadOrCropD(spatial_size=INPUT_SHAPE_AE, mode='minimum', keys=['image']),
    ScaleIntensityD(minv=0, maxv=1, keys=['image']),
])


def preprocess_mri(image_path: str, device: str = "cpu") -> torch.Tensor:
    """
    Preprocess an MRI using MONAI transforms to produce
    a 5D tensor (batch=1, channel=1, D, H, W) for inference.

    Args:
        image_path (str): Path to the MRI (e.g. .nii.gz).
        device (str): Device to place the tensor on.

    Returns:
        torch.Tensor: Shape (1, 1, D, H, W).
    """
    data_dict = {"image_path": image_path}
    output_dict = transforms_fn(data_dict)
    image_tensor = output_dict["image"]  # shape: (1, D, H, W)
    image_tensor = image_tensor.unsqueeze(0)  # => (1, 1, D, H, W)
    return image_tensor.to(device)


class Brain2vec(AutoencoderKL):
    """
    Subclass of MONAI's AutoencoderKL that includes:
      - a from_pretrained(...) for loading a .pth checkpoint
      - uses the existing forward(...) that returns (reconstruction, z_mu, z_sigma)

    Usage:
      >>> model = Brain2vec.from_pretrained("my_checkpoint.pth", device="cuda")
      >>> image_tensor = preprocess_mri("/path/to/mri.nii.gz", device="cuda")
      >>> reconstruction, z_mu, z_sigma = model.forward(image_tensor)
    """

    @staticmethod
    def from_pretrained(
        checkpoint_path: Optional[str] = None,
        device: str = "cpu"
    ) -> nn.Module:
        """
        Load a pretrained Brain2vec (AutoencoderKL) if a checkpoint_path is provided.
        Otherwise, return an uninitialized model.

        Args:
            checkpoint_path (Optional[str]): Path to a .pth checkpoint file.
            device (str): "cpu", "cuda", "mps", etc.

        Returns:
            nn.Module: The loaded Brain2vec model on the chosen device.
        """
        model = Brain2vec(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            latent_channels=1,
            num_channels=(64, 128, 128, 128),
            num_res_blocks=2,
            norm_num_groups=32,
            norm_eps=1e-06,
            attention_levels=(False, False, False, False),
            with_decoder_nonlocal_attn=False,
            with_encoder_nonlocal_attn=False,
        )

        if checkpoint_path is not None:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found.")
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict)

        model.to(device)
        model.eval()
        return model


def main() -> None:
    """
    Main function to parse command-line arguments and run inference
    with a pretrained Brain2vec model.
    """
    parser = argparse.ArgumentParser(
        description="Inference script for a Brain2vec (VAE) model."
    )
    parser.add_argument(
        "--checkpoint_path", type=str, required=True,
        help="Path to the .pth checkpoint of the pretrained Brain2vec model."
    )
    parser.add_argument(
        "--output_dir", type=str, default="./vae_inference_outputs",
        help="Directory to save reconstructions and latent parameters."
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device to run inference on ('cpu', 'cuda', etc.)."
    )
    # Two ways to supply images: multiple file paths or a CSV
    parser.add_argument(
        "--input_images", type=str, nargs="*",
        help="One or more MRI file paths (e.g. .nii.gz)."
    )
    parser.add_argument(
        "--csv_input", type=str,
        help="Path to a CSV file with an 'image_path' column."
    )
    parser.add_argument(
        "--embeddings_filename",
        type=str,
        required=True,
        help="Filename (in output_dir) to save the stacked z_mu embeddings (e.g. 'all_z_mu.npy')."
    )
    parser.add_argument(
        "--save_recons",
        action="store_true",
        help="If set, saves each reconstruction as .npy. Default is not to save."
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load the model
    model = Brain2vec.from_pretrained(
        checkpoint_path=args.checkpoint_path,
        device=args.device
    )

    # Gather image paths
    if args.csv_input:
        df = pd.read_csv(args.csv_input)
        if "image_path" not in df.columns:
            raise ValueError("CSV must contain a column named 'image_path'.")
        image_paths = df["image_path"].tolist()
    else:
        if not args.input_images:
            raise ValueError("Must provide either --csv_input or --input_images.")
        image_paths = args.input_images

    # Lists for stacking latent parameters later
    all_z_mu = []
    all_z_sigma = []

    # Inference on each image
    for i, img_path in enumerate(image_paths):
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        print(f"[INFO] Processing image {i}: {img_path}")
        img_tensor = preprocess_mri(img_path, device=args.device)

        with torch.no_grad():
            recon, z_mu, z_sigma = model.forward(img_tensor)

        # Convert to NumPy
        recon_np = recon.detach().cpu().numpy()  # shape: (1, 1, D, H, W)
        z_mu_np = z_mu.detach().cpu().numpy()    # shape: (1, latent_channels, ...)
        z_sigma_np = z_sigma.detach().cpu().numpy()

        # Save each reconstruction (per image) as .npy
    if args.save_recons:
        recon_path = os.path.join(args.output_dir, f"reconstruction_{i}.npy")
        np.save(recon_path, recon_np)
        print(f"[INFO] Saved reconstruction to {recon_path}")

        # Store latent parameters for optional combined saving
        all_z_mu.append(z_mu_np)
        all_z_sigma.append(z_sigma_np)

    # Combine latent parameters from all images and save
    stacked_mu = np.concatenate(all_z_mu, axis=0)       # e.g., shape (N, latent_channels, ...)
    stacked_sigma = np.concatenate(all_z_sigma, axis=0) # e.g., shape (N, latent_channels, ...)

    mu_filename = args.embeddings_filename
    if not mu_filename.lower().endswith(".npy"):
        mu_filename += ".npy"

    mu_path = os.path.join(args.output_dir, mu_filename)
    sigma_path = os.path.join(args.output_dir, "all_z_sigma.npy")

    np.save(mu_path, stacked_mu)
    np.save(sigma_path, stacked_sigma)

    print(f"[INFO] Saved z_mu of shape {stacked_mu.shape} to {mu_path}")
    print(f"[INFO] Saved z_sigma of shape {stacked_sigma.shape} to {sigma_path}")


if __name__ == "__main__":
    main()