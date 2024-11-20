#!/bin/bash
#SBATCH -J python_sma_grad_inner_argmax
#SBATCH -p node
#SBATCH -t 72:00:00
#SBATCH -N 1
#SBATCH -n 48

module load intel18u4

ulimit -u unlimited
ulimit -m unlimited
ulimit -s unlimited

date
conda run -n SMA_grad_inner_argmax python3 bo_botorch_grad_opt.py
date