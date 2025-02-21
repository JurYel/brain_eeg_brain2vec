---
license: apache-2.0
language:
  - en
task_categories:
  - image-classification
tags:
  - medical
  - brain-data
  - mri
pretty_name: 3D Brain Structure MRI Autoencoder
---

## 🧠 Model Summary
# brain2vec
An autoencoder model for brain structure T1 MRIs based on [Brain Latent Progression](https://github.com/LemuelPuglisi/BrLP/tree/main). The autoencoder takes in a 3d MRI NIfTI file and compresses to 1200 latent dimensions before reconstructing the image. The loss functions for training the autoencoder are:
- [L1Loss](https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html)
- [KLDivergenceLoss](https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html)
- [PatchAdversarialLoss](https://docs.monai.io/en/stable/losses.html#patchadversarialloss)
- [PerceptualLoss](https://docs.monai.io/en/stable/losses.html#perceptualloss)



# Training data
[Radiata brain-structure](https://huggingface.co/datasets/radiata-ai/brain-structure): 3066 scans from 2085 individuals in the 'train' split. Mean age = 45.1 +- 24.5, including 2847 scans from cognitively normal subjects and 219 scans from individuals with an Alzheimer's disease clinical diagnosis.

# Example usage
```
# get brain2vec model repository
git clone https://huggingface.co/radiata-ai/brain2vec
cd brain2vec

# set up virtual environemt
python3 -m venv venv_brain2vec
source venv_brain2vec/bin/activate

# install Python libraries
pip install -r requirements.txt

# create the csv file inputs.csv listing the scan paths and other info
# this script loads the radiata-ai/brain-structure dataset
python create_csv.py

mkdir ae_cache
mkdir ae_output

# train the model
nohup python brain2vec.py train \
  --dataset_csv /home/ubuntu/brain2vec/inputs.csv \
  --cache_dir   ./ae_cache \
  --output_dir  ./ae_output \
  --n_epochs    10 \
> train_log.txt 2>&1 &

# model inference
python inference_brain2vec.py \
  --checkpoint_path /path/to/model.pth \
  --input_images /path/to/img1.nii.gz /path/to/img2.nii.gz \
  --output_dir ./vae_inference_outputs \
  --embeddings_filename pca_output/pca_embeddings_2.npy \
  --save_recons
```

# Methods
Input scan image dimensions are 113x137x113, 1.5mm^3 resolution, aligned to MNI152 space (see [radiata-ai/brain-structure](https://huggingface.co/datasets/radiata-ai/brain-structure)).  

The image transform crops to 80 x 96 x 80, 2mm^3 resolution, and scales image intensity to range [0,1]. Images are flattened to 614400-length 1D vectors.  

10 epochs
    max_batch_size: int = 2,
    batch_size: int = 16,
    lr: float = 1e-4,

# References
Puglisi
Pinaya

# Citation
```
@misc{Radiata-Brain2Vec,
  author    = {Jesse Brown and Clayton Young},
  title     = {brain2vec_PCA: A VAE Model for Brain Structure T1 MRIs},
  year      = {2025},
  url       = {https://huggingface.co/radiata-ai/brain2vec},
  note      = {Version 1.0},
  publisher = {Hugging Face}
}
```

# License
MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.