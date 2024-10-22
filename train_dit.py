import torch 
import time
import os
import argparse
import shutil
import sys
 
def parse_args():
    parser = argparse.ArgumentParser(description='Matrix multiplication')
    parser.add_argument('--gpus', help='gpu amount', default='0,1,2,3,4,5,6,7', type=str)
    parser.add_argument('--size', help='matrix size', default='52000', type=int)
    parser.add_argument('--interval', help='sleep interval', default='0.025', type=float)
    args = parser.parse_args()
    return args
 
 
def matrix_multiplication(args):
 
    a_list, b_list, result = [], [], []    
    size = (args.size, args.size)
    gpu_ids = [int(gpu_id) for gpu_id in args.gpus.split(',')]
    
    for i in gpu_ids:
        a_list.append(torch.rand(size, device=f'cuda:{i}'))
        b_list.append(torch.rand(size, device=f'cuda:{i}'))
        result.append(torch.rand(size, device=f'cuda:{i}'))
 
    while True:
        for i in range(len(gpu_ids)):
            result[i] = a_list[i] * b_list[i]
        time.sleep(args.interval)
 
if __name__ == "__main__":
    # usage:   python gpuhold.py  --size 38000 --gpus 4,5,6,7 --interval 0.01
    args = parse_args()
    matrix_multiplication(args)
    
    
    
    
