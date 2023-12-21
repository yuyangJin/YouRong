#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 srun -n1 --exclusive=user  --gres=gpu:8 -p Big -u  python3 ./examples/train.py 
