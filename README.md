# SPARSE-GUARD: 
SPARSE Coding based defense against Reconstruction Attacks

This repo contains experiment on CIFAR10, MNIST, Fashion MNIST datasets

Each of the 3 directories (CIFAR10, MNIST, and FMNIST) contains 4 files for 4 different target models on Split Learning. The target model that achieves lower reconstruction performance (i.e., lower PSNR, lower SSIM, and higher FID), means it works as better defense against the reconstruction attack. 

For example, mnist_cnn.py contains the code to train CNN architecture-based target model, performs the attack to reconstruct all training instances on MNIST dataset and finally computes the PSNR, SSIM, and FID scores using the original training sample and reconstructed sample using the model inversion attack against the split neural network. 


Each of the 3 directories also contain another 4 python code files starting with *etn* prefix to denote the similar attacks against that particular dataset using the end-to-end network, where the adversary can only access the output just before the classification layer.

distribution directory contains the plots on entire dataset as well as single image across different models

Dist_plot.ipynb is for plotting the distributions (data files for this code can be generated using the code files provided in the *other* directory)

model_attack and model_target directories contains different models that are trained in our codes

To run a code file first one has to install conda environment, pytorch, and other required packages.
Once all installation is complete, one can run the following commands top activate the conda env and finally run the shell script (i.e., test.sh) provided to execute a python code file.

```
####module load miniconda3
####source activate your_env
####sbatch test.sh
```

# License
Sparse-Guard is provided under a BSD license with a "modifications must be indicated" clause. See the LICENSE file for the full text. Sparse-Guard is covered under the LCA-PyTorch package, known internally as LA-CC-23-064.
