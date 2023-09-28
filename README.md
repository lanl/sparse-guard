© 2023. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
Department of Energy/National Nuclear Security Administration.

# SPARSE-GUARD: 
SPARSE Coding based defense against Reconstruction Attacks

This repo contains experiments on MNIST, Fashion MNIST datasets using our proposed SPARSE-GUARD and standard benchmarks.

Each of the 2 directories (MNIST, and FMNIST) contains files to implement and test different SOTA defenses as well as our proposed SPARSE-GUARD on both Split Network as well as End-to-end Networks. The defense that achieves poor reconstruction performance (i.e., lower PSNR, lower SSIM, and higher FID), indicate it as better defense against the reconstruction attacks, i.e., model inversion attack. 

For example, mnist_gaussiannoise.py contains the code to train defense with target model using Gaussian Noise, then performs the attack to reconstruct all training instances on MNIST dataset and finally computes the PSNR, SSIM, and FID scores using the original training sample and reconstructed sample. 


Each of the directories also contain another sets of python code files starting with *etn* prefix to denote the similar attacks against that particular dataset using the end-to-end network, where the adversary can only access the output of last hidden layer before the classification layer.

"other" directory contains code to plot the Umap representation of linear, convolution and sparse coding layers.


"model" directory contains different saved models that are trained with our codes

***For lambda value experiment, vary the lambda value in corresponding code file.

To run a code file first one has to install conda environment, pytorch, and other required packages.
Once all installation is complete, one can run the following commands top activate the conda env and finally run the shell script (i.e., test.sh) provided to execute a python code file.

```
####module load miniconda3
####source activate your_env
####sbatch test.sh
```

# License
Sparse-Guard is provided under a BSD license with a "modifications must be indicated" clause. See the LICENSE file for the full text. Sparse-Guard is covered under the LCA-PyTorch package, known internally as LA-CC-23-064.
