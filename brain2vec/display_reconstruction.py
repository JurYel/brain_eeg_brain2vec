import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize

def display_original_and_reconstruction(original_fpath, reconstructed_fpath, n=10):

    # Load the original and reconstructed images
    original = np.load(original_fpath)
    reconstructed = np.load(reconstructed_fpath)

    # Remove singleton dimensions -> (Depth, Height, Width)
    original = original.squeeze() # Shape: (10, 12, 10)
    reconstructed = reconstructed.squeeze() # Shape: (80, 96, 80)

    # Resize original to match the reconstructed dimensions
    # original_resized = resize(original, reconstructed.shape, mode="reflect", anti_aliasing=True)

    # print(original_resized.shape)   

    # Select the middle slices along the depth axis
    # mid_slice_original = original_resized[original_resized.shape[0] // 2, :, :] # Shape: (12, 10)
    mid_slice_original = original[original.shape[0] // 2, :, :] # Shape: (12, 10)
    mid_slice_reconstructed = reconstructed[reconstructed.shape[0] // 2, :, :] # Shape: (96, 80)

    # Use subplots to display side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Show original image
    axes[0].imshow(mid_slice_original, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Show reconstructed image
    axes[1].imshow(mid_slice_reconstructed, cmap="gray")
    axes[1].set_title("Reconstructed Image")
    axes[1].axis("off")

    plt.tight_layout()
    # plt.title("Original vs Reconstructed Image ", plt.gcf())
    # plt.title("Original vs Reconstructed MRI Image ")
    plt.show()

if __name__ == "__main__":
    original_fpath = "./vae_inference_outputs/all_z_sigma.npy"
    reconstructed_fpath = "./vae_inference_outputs/reconstruction_1.npy"
    display_original_and_reconstruction(original_fpath, reconstructed_fpath)