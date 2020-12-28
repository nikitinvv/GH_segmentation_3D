#!/bin/bash
#SBATCH -t 2-00:00:00
#SBATCH -J rec
#SBATCH --mem 160G
#SBATCH --exclusive
#SBATCH -N 1
#SBATCH --tasks-per-node=40
#SBATCH -p v100
#SBATCH --exclude gn1
module add GCC/8.3.0 GCCcore/8.3.0  CUDA/10.1.243 OpenMPI/3.1.4 LibTIFF/4.0.10 scikit-image/0.16.2-Python-3.7.4  TensorFlow/2.1.0-Python-3.7.4

python -c 'import tensorflow'
python unet3D_new.py