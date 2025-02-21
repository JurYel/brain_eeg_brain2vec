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
pretty_name: 3D Brain Structure MRI PCA
---

## 🧠 Model Summary
# brain2vec_PCA
A linear PCA model for brain structure T1 MRIs. The models takes in a 3d MRI NIfTI file and compresses to 1200 latent dimensions before reconstructing the image.


# Training data
[radiata-ai/brain-structure](https://huggingface.co/datasets/radiata-ai/brain-structure): 3066 scans from 2085 individuals in the 'train' split. Mean age = 45.1 +- 24.5, including 2847 scans from cognitively normal subjects and 219 scans from individuals with an Alzheimer's disease clinical diagnosis.

# Example usage
```
# get brain2vec_PCA model repository
git lfs install  # Ensure Git LFS is installed and enabled
git clone https://huggingface.co/radiata-ai/brain2vec_PCA
cd brain2vec_PCA
git lfs pull

# set up virtual environemt
python3 -m venv venv_brain2vec_PCA
source venv_brain2vec_PCA/bin/activate

# install Python libraries
pip install -r requirements.txt

# create the csv file inputs.csv listing the scan paths and other info
# this script loads the radiata-ai/brain-structure dataset from Hugging Face
python create_csv.py

mkdir pca_output

# train the model
nohup python train_brain2vec_PCA.py \
    --inputs_csv inputs.csv \
    --output_dir ./pca_output \
    --pca_type standard \
    --n_components 1200 \
    > train_log.txt 2>&1 &

# model inference
python inference_brain2vec_PCA.py \
    --pca_model pca_model.joblib \
    --input_images /path/to/img1.nii.gz /path/to/img2.nii.gz \
    --output_dir pca_output \
    --embeddings_filename pca_embeddings_2 \
    --save_recons

python inference_brain2vec_PCA.py \
    --pca_model pca_model.joblib \
    --csv_input inputs.csv \
    --output_dir pca_output \
    --embeddings_filename pca_embeddings_all
```

# Methods
Input scan image dimensions are 113x137x113, 1.5mm^3 resolution, aligned to MNI152 space (see [radiata-ai/brain-structure](https://huggingface.co/datasets/radiata-ai/brain-structure)).  

The image transform crops to 80 x 96 x 80, 2mm^3 resolution, and scales image intensity to range [0,1]. Images are flattened to 614400-length 1D vectors.

PCA is performed using [sklearn.decomposition.PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).  


# Citation
```
@misc{Radiata-Brain2Vec-PCA,
  author    = {Jesse Brown and Clayton Young},
  title     = {brain2vec_PCA: A Linear PCA Model for Brain Structure T1 MRIs},
  year      = {2025},
  url       = {https://huggingface.co/radiata-ai/brain2vec_PCA},
  note      = {Version 1.0},
  publisher = {Hugging Face}
}
```

# License
### Apache License 2.0

Copyright 2025 Jesse Brown

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at:

[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
