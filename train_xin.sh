#!/bin/bash
#SBATCH --job-name=llama-lora
#SBATCH --partition=es1
#SBATCH --qos=es_lowprio
#SBATCH --account=pc_automat
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --constraint=es1_a40
#SBATCH --time=10:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=chenxin0210@lbl.gov
#export CUDA_VISIBLE_DEVICES=0
echo current conda env is $CONDA_DEFAULT_ENV
echo "================"
#echo current GPU condition is:
#python gpu.py
echo available nCPU is:
nproc

nvidia-smi
echo "================"
echo start running:

accelerate launch --mixed_precision fp16 run-llama.py
