#!/usr/bin/env python3

"""
train_brain2vec.py

Trains a 3D VAE-based Brain2Vec model using MONAI. This script implements 
autoencoder training with adversarial loss (via a patch discriminator),
a perceptual loss, and KL divergence regularization for robust latent 
representations. 

Example usage:
    python train_brain2vec.py train \
        --dataset_csv /path/to/dataset.csv \
        --cache_dir /path/to/cache \
        --output_dir /path/to/output_dir \
        --n_epochs 10
"""

import os
os.environ["PYTORCH_WEIGHTS_ONLY"] = "False"
from typing import Optional, Union
import pandas as pd
import argparse
import numpy as np
import warnings
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.nn import L1Loss
from torch.utils.data import DataLoader
from torch.amp import autocast
from torch.amp import GradScaler
from generative.networks.nets import (
    AutoencoderKL, 
    PatchDiscriminator,
)
from generative.losses import PerceptualLoss, PatchAdversarialLoss
from monai.data import Dataset, PersistentDataset
from monai.transforms.transform import Transform
from monai import transforms
from monai.utils import set_determinism
from monai.data.meta_tensor import MetaTensor
import torch.serialization
from numpy.core.multiarray import _reconstruct
from numpy import ndarray, dtype
torch.serialization.add_safe_globals([_reconstruct])
torch.serialization.add_safe_globals([MetaTensor])
torch.serialization.add_safe_globals([ndarray])
torch.serialization.add_safe_globals([dtype])
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# voxel resolution
RESOLUTION = 2                              

# shape of the MNI152 (1mm^3) template
INPUT_SHAPE_1mm = (182, 218, 182)   

# resampling the MNI152 to (1.5mm^3)
INPUT_SHAPE_1p5mm = (122, 146, 122)   

# Adjusting the dimensions to be divisible by 8 (2^3 where 3 are the downsampling layers of the AE)
#INPUT_SHAPE_AE = (120, 144, 120)
INPUT_SHAPE_AE = (80, 96, 80)

# Latent shape of the autoencoder 
LATENT_SHAPE_AE = (1, 10, 12, 10)


def load_if(checkpoints_path: Optional[str], network: nn.Module) -> nn.Module:
    """
    Load pretrained weights if available.

    Args:
        checkpoints_path (Optional[str]): path of the checkpoints
        network (nn.Module): the neural network to initialize 

    Returns:
        nn.Module: the initialized neural network
    """
    if checkpoints_path is not None:
        assert os.path.exists(checkpoints_path), 'Invalid path'
        network.load_state_dict(torch.load(checkpoints_path))
    return network


def init_autoencoder(checkpoints_path: Optional[str] = None) -> nn.Module:
    """
    Load the KL autoencoder (pretrained if `checkpoints_path` points to previous params).

    Args:
        checkpoints_path (Optional[str], optional): path of the checkpoints. Defaults to None.

    Returns:
        nn.Module: the KL autoencoder
    """
    autoencoder = AutoencoderKL(spatial_dims=3, 
                                in_channels=1, 
                                out_channels=1, 
                                latent_channels=1, #3,
                                num_channels=(64, 128, 128, 128),
                                num_res_blocks=2, 
                                norm_num_groups=32,
                                norm_eps=1e-06,
                                attention_levels=(False, False, False, False), 
                                with_decoder_nonlocal_attn=False, 
                                with_encoder_nonlocal_attn=False)
    return load_if(checkpoints_path, autoencoder)


def init_patch_discriminator(checkpoints_path: Optional[str] = None) -> nn.Module:
    """
    Load the patch discriminator (pretrained if `checkpoints_path` points to previous params).

    Args:
        checkpoints_path (Optional[str], optional): path of the checkpoints. Defaults to None.

    Returns:
        nn.Module: the patch discriminator
    """
    patch_discriminator = PatchDiscriminator(spatial_dims=3, 
                                             num_layers_d=3, 
                                             num_channels=32, 
                                             in_channels=1, 
                                             out_channels=1)
    return load_if(checkpoints_path, patch_discriminator)


class KLDivergenceLoss:
    """
    A class for computing the Kullback-Leibler divergence loss.
    """
    
    def __call__(self, z_mu: Tensor, z_sigma: Tensor) -> Tensor:
        """
        Computes the KL divergence loss for the given parameters.

        Args:
            z_mu (Tensor):  The mean of the distribution.
            z_sigma (Tensor): The standard deviation of the distribution.

        Returns:
            Tensor: The computed KL divergence loss, averaged over the batch size.
        """

        kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
        return torch.sum(kl_loss) / kl_loss.shape[0]


class GradientAccumulation:
    """
    Implements gradient accumulation to facilitate training with larger 
    effective batch sizes than what can be physically accommodated in memory.
    """

    def __init__(self,
                 actual_batch_size: int, 
                 expect_batch_size: int,
                 loader_len: int,
                 optimizer: Optimizer, 
                 grad_scaler: Optional[GradScaler] = None) -> None:
        """
        Initializes the GradientAccumulation instance with the necessary parameters for 
        managing gradient accumulation.

        Args:
            actual_batch_size (int): The size of the mini-batches actually used in training.
            expect_batch_size (int): The desired (effective) batch size to simulate through gradient accumulation.
            loader_len (int): The length of the data loader, representing the total number of mini-batches.
            optimizer (Optimizer): The optimizer used for performing optimization steps.
            grad_scaler (Optional[GradScaler], optional): A GradScaler for mixed precision training. Defaults to None.
        
        Raises:
            AssertionError: If `expect_batch_size` is not divisible by `actual_batch_size`.
        """

        assert expect_batch_size % actual_batch_size == 0, \
            'expect_batch_size must be divisible by actual_batch_size'
        self.actual_batch_size = actual_batch_size
        self.expect_batch_size = expect_batch_size
        self.loader_len = loader_len
        self.optimizer = optimizer
        self.grad_scaler = grad_scaler

        # if the expected batch size is N=KM, and the actual batch size
        # is M, then we need to accumulate gradient from N / M = K optimization steps. 
        self.steps_until_update = expect_batch_size / actual_batch_size

    def step(self, loss: Tensor, step: int) -> None:
        """
        Performs a backward pass for the given loss and potentially executes an optimization 
        step if the conditions for gradient accumulation are met. The optimization step is taken 
        only after a specified number of steps (defined by the expected batch size) or at the end 
        of the dataset.

        Args:
            loss (Tensor): The loss value for the current forward pass.
            step (int): The current step (mini-batch index) within the epoch.
        """
        loss = loss / self.expect_batch_size
        
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
        else:
            loss.backward()        
        if (step + 1) % self.steps_until_update == 0 or (step + 1) == self.loader_len:
            if self.grad_scaler is not None:
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)


class AverageLoss:
    """
    Utility class to track losses
    and metrics during training.
    """

    def __init__(self):
        self.losses_accumulator = {}
    
    def put(self, loss_key:str, loss_value:Union[int,float]) -> None:
        """
        Store value

        Args:
            loss_key (str): Metric name
            loss_value (int | float): Metric value to store
        """
        if loss_key not in self.losses_accumulator:
            self.losses_accumulator[loss_key] = []
        self.losses_accumulator[loss_key].append(loss_value)
    
    def pop_avg(self, loss_key:str) -> float:
        """
        Average the stored values of a given metric

        Args:
            loss_key (str): Metric name

        Returns:
            float: average of the stored values
        """
        if loss_key not in self.losses_accumulator:
            return None
        losses = self.losses_accumulator[loss_key]
        self.losses_accumulator[loss_key] = []
        return sum(losses) / len(losses)
    
    def to_tensorboard(self, writer: SummaryWriter, step: int):
        """
        Logs the average value of all the metrics stored 
        into Tensorboard.

        Args:
            writer (SummaryWriter): Tensorboard writer
            step (int): Tensorboard logging global step 
        """
        for metric_key in self.losses_accumulator.keys():
            writer.add_scalar(metric_key, self.pop_avg(metric_key), step)


def get_dataset_from_pd(df: pd.DataFrame, transforms_fn: Transform, cache_dir: Optional[str]) -> Union[Dataset,PersistentDataset]: 
    """
    If `cache_dir` is defined, returns a `monai.data.PersistenDataset`. 
    Otherwise, returns a simple `monai.data.Dataset`.

    Args:
        df (pd.DataFrame): Dataframe describing each image in the longitudinal dataset.
        transforms_fn (Transform): Set of transformations
        cache_dir (Optional[str]): Cache directory (ensure enough storage is available)

    Returns:
        Dataset|PersistentDataset: The dataset
    """
    assert cache_dir is None or os.path.exists(cache_dir), 'Invalid cache directory path'
    data = df.to_dict(orient='records')
    return Dataset(data=data, transform=transforms_fn) if cache_dir is None \
        else PersistentDataset(data=data, transform=transforms_fn, cache_dir=cache_dir)


def tb_display_reconstruction(writer, step, image, recon):
    """
    Display reconstruction in TensorBoard during AE training.
    """
    plt.style.use('dark_background')
    _, ax = plt.subplots(ncols=3, nrows=2, figsize=(7, 5))
    for _ax in ax.flatten(): _ax.set_axis_off()

    if len(image.shape) == 4: image = image.squeeze(0) 
    if len(recon.shape) == 4: recon = recon.squeeze(0)

    ax[0, 0].set_title('original image', color='cyan')
    ax[0, 0].imshow(image[image.shape[0] // 2, :, :], cmap='gray')
    ax[0, 1].imshow(image[:, image.shape[1] // 2, :], cmap='gray')
    ax[0, 2].imshow(image[:, :, image.shape[2] // 2], cmap='gray')

    ax[1, 0].set_title('reconstructed image', color='magenta')
    ax[1, 0].imshow(recon[recon.shape[0] // 2, :, :], cmap='gray')
    ax[1, 1].imshow(recon[:, recon.shape[1] // 2, :], cmap='gray')
    ax[1, 2].imshow(recon[:, :, recon.shape[2] // 2], cmap='gray')

    plt.tight_layout()
    writer.add_figure('Reconstruction', plt.gcf(), global_step=step)


def set_environment(seed: int = 0) -> None:
    """
    Set deterministic behavior for reproducibility.

    Args:
        seed (int, optional): Seed value. Defaults to 0.
    """
    set_determinism(seed)


def train(
    dataset_csv: str,
    cache_dir: str,
    output_dir: str,
    aekl_ckpt: Optional[str] = None,
    disc_ckpt: Optional[str] = None,
    num_workers: int = 8,
    n_epochs: int = 5,
    max_batch_size: int = 2,
    batch_size: int = 16,
    lr: float = 1e-4,
    aug_p: float = 0.8,
    device: str = ('cuda' if torch.cuda.is_available() else 
                   'cpu'),
) -> None:
    """
    Train the autoencoder and discriminator models.

    Args:
        dataset_csv (str): Path to the dataset CSV file.
        cache_dir (str): Directory for caching data.
        output_dir (str): Directory to save model checkpoints.
        aekl_ckpt (Optional[str], optional): Path to the autoencoder checkpoint. Defaults to None.
        disc_ckpt (Optional[str], optional): Path to the discriminator checkpoint. Defaults to None.
        num_workers (int, optional): Number of data loader workers. Defaults to 8.
        n_epochs (int, optional): Number of training epochs. Defaults to 5.
        max_batch_size (int, optional): Actual batch size per iteration. Defaults to 2.
        batch_size (int, optional): Expected (effective) batch size. Defaults to 16.
        lr (float, optional): Learning rate. Defaults to 1e-4.
        aug_p (float, optional): Augmentation probability. Defaults to 0.8.
        device (str, optional): Device to run the training on. Defaults to 'cuda' if available.
    """
    set_environment(0)

    transforms_fn = transforms.Compose([
        transforms.CopyItemsD(keys={'image_path'}, names=['image']),
        transforms.LoadImageD(image_only=True, keys=['image']),
        transforms.EnsureChannelFirstD(keys=['image']),
        transforms.SpacingD(pixdim=2, keys=['image']),
        transforms.ResizeWithPadOrCropD(spatial_size=(80, 96, 80), mode='minimum', keys=['image']),
        transforms.ScaleIntensityD(minv=0, maxv=1, keys=['image'])
    ])

    dataset_df = pd.read_csv(dataset_csv)
    train_df = dataset_df[dataset_df.split == 'train']
    trainset = get_dataset_from_pd(train_df, transforms_fn, cache_dir)

    train_loader = DataLoader(
        dataset=trainset,
        num_workers=num_workers,
        batch_size=max_batch_size,
        shuffle=True,
        persistent_workers=True,
        pin_memory=True,
    )

    print('Device is %s' %(device))
    autoencoder = init_autoencoder(aekl_ckpt).to(device)
    discriminator = init_patch_discriminator(disc_ckpt).to(device)

    # Loss Weights
    adv_weight = 0.025
    perceptual_weight = 0.001
    kl_weight = 1e-7

    # Loss Functions
    l1_loss_fn = L1Loss()
    kl_loss_fn = KLDivergenceLoss()
    adv_loss_fn = PatchAdversarialLoss(criterion="least_squares")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        perc_loss_fn = PerceptualLoss(
            spatial_dims=3,
            network_type="squeeze",
            is_fake_3d=True,
            fake_3d_ratio=0.2
        ).to(device)

    # Optimizers
    optimizer_g = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr)

    # Gradient Accumulation
    gradacc_g = GradientAccumulation(
        actual_batch_size=max_batch_size,
        expect_batch_size=batch_size,
        loader_len=len(train_loader),
        optimizer=optimizer_g,
        grad_scaler=GradScaler()
    )

    gradacc_d = GradientAccumulation(
        actual_batch_size=max_batch_size,
        expect_batch_size=batch_size,
        loader_len=len(train_loader),
        optimizer=optimizer_d,
        grad_scaler=GradScaler()
    )

    # Logging
    avgloss = AverageLoss()
    writer = SummaryWriter()
    total_counter = 0

    for epoch in range(n_epochs):
        print(f"[DEBUG] Starting epoch {epoch}/{n_epochs-1}")
        autoencoder.train()
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        progress_bar.set_description(f'Epoch {epoch}')

        for step, batch in progress_bar:
            # Generator Training
            with autocast(device, enabled=True):
                images = batch["image"].to(device)
                reconstruction, z_mu, z_sigma = autoencoder(images)

                logits_fake = discriminator(reconstruction.contiguous().float())[-1]

                rec_loss = l1_loss_fn(reconstruction.float(), images.float())
                kl_loss = kl_weight * kl_loss_fn(z_mu, z_sigma)
                per_loss = perceptual_weight * perc_loss_fn(reconstruction.float(), images.float())
                gen_loss = adv_weight * adv_loss_fn(logits_fake, target_is_real=True, for_discriminator=False)

                loss_g = rec_loss + kl_loss + per_loss + gen_loss

            gradacc_g.step(loss_g, step)

            # Discriminator Training
            with autocast(device, enabled=True):
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                d_loss_fake = adv_loss_fn(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                d_loss_real = adv_loss_fn(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (d_loss_fake + d_loss_real) * 0.5
                loss_d = adv_weight * discriminator_loss

            gradacc_d.step(loss_d, step)

            # Logging
            avgloss.put('Generator/reconstruction_loss', rec_loss.item())
            avgloss.put('Generator/perceptual_loss', per_loss.item())
            avgloss.put('Generator/adversarial_loss', gen_loss.item())
            avgloss.put('Generator/kl_regularization', kl_loss.item())
            avgloss.put('Discriminator/adversarial_loss', loss_d.item())

            if total_counter % 10 == 0:
                step_log = total_counter // 10
                avgloss.to_tensorboard(writer, step_log)
                tb_display_reconstruction(
                    writer,
                    step_log,
                    images[0].detach().cpu(),
                    reconstruction[0].detach().cpu()
                )

            total_counter += 1

        # Save the model after each epoch.
        os.makedirs(output_dir, exist_ok=True)
        torch.save(discriminator.state_dict(), os.path.join(output_dir, f'discriminator-ep-{epoch}.pth'))
        torch.save(autoencoder.state_dict(), os.path.join(output_dir, f'autoencoder-ep-{epoch}.pth'))

    writer.close()
    print("Training completed and models saved.")


def main():
    """
    Main function to parse command-line arguments and execute training.
    """
    parser = argparse.ArgumentParser(description="brain2vec Training Script")

    subparsers = parser.add_subparsers(dest='command', required=True, help='Sub-commands: train or infer')

    # Training Subparser
    train_parser = subparsers.add_parser('train', help='Train the models.')
    train_parser.add_argument('--dataset_csv', type=str, required=True, help='Path to the dataset CSV file.')
    train_parser.add_argument('--cache_dir', type=str, required=True, help='Directory for caching data.')
    train_parser.add_argument('--output_dir', type=str, required=True, help='Directory to save model checkpoints.')
    train_parser.add_argument('--aekl_ckpt', type=str, default=None, help='Path to the autoencoder checkpoint.')
    train_parser.add_argument('--disc_ckpt', type=str, default=None, help='Path to the discriminator checkpoint.')
    train_parser.add_argument('--num_workers', type=int, default=8, help='Number of data loader workers.')
    train_parser.add_argument('--n_epochs', type=int, default=5, help='Number of training epochs.')
    train_parser.add_argument('--max_batch_size', type=int, default=2, help='Actual batch size per iteration.')
    train_parser.add_argument('--batch_size', type=int, default=16, help='Expected (effective) batch size.')
    train_parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    train_parser.add_argument('--aug_p', type=float, default=0.8, help='Augmentation probability.')

    args = parser.parse_args()

    if args.command == 'train':
        train(
            dataset_csv=args.dataset_csv,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            aekl_ckpt=args.aekl_ckpt,
            disc_ckpt=args.disc_ckpt,
            num_workers=args.num_workers,
            n_epochs=args.n_epochs,
            max_batch_size=args.max_batch_size,
            batch_size=args.batch_size,
            lr=args.lr,
            aug_p=args.aug_p,
        )
    elif args.command == 'infer':
        inference(
            dataset_csv=args.dataset_csv,
            aekl_ckpt=args.aekl_ckpt,
            output_dir=args.output_dir,
        )
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
