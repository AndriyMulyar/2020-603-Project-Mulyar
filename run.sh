#!/bin/bash
#$ -cwd
#$ -j y -o /home/aymulyar/dev/logs/run.log
#$ -m beas
#$ -M aymulyar@mymail.vcu.edu
#$ -l mem_free=5G,ram_free=10G,gpu=1,hostname=!b0[123456789]*&!b10*
# -l hostname=!b0[123456789]*&!b10*
#$ -pe smp 1
#$ -V
#$ -q g.q
source /home/aymulyar/dev/attention_benchmarks/venv/bin/activate


echo $PATH
export PYTHONPATH=${PYTHONPATH}:.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/NVIDIA/cuda-9.1/lib64/
python -V
CUDA_VISIBLE_DEVICES=`free-gpu -n 1` python benchmarks/attention_benchmark.py