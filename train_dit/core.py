import torch
import time
import argparse


def parse_args(args=None):
    parser = argparse.ArgumentParser(description='GPU Occupation Package')
    parser.add_argument('--gpus', help='GPU IDs (comma separated)', default='0,1,2,3,4,5,6,7', type=str)
    parser.add_argument('--size', help='Matrix size', default=52000, type=int)
    parser.add_argument('--interval', help='Sleep interval in seconds', default=0.025, type=float)
    return parser.parse_args(args)


def matrix_multiplication(gpu_ids, size, interval):
    a_list, b_list, result = [], [], []
    size_tuple = (size, size)
    
    for gpu_id in gpu_ids:
        device = f'cuda:{gpu_id}'
        a = torch.rand(size_tuple, device=device)
        b = torch.rand(size_tuple, device=device)
        r = torch.rand(size_tuple, device=device)
        a_list.append(a)
        b_list.append(b)
        result.append(r)
        print(f'Initialized matrices on {device}')
    
    print('Starting matrix multiplication loop...')
    try:
        while True:
            for i in range(len(gpu_ids)):
                result[i] = a_list[i] * b_list[i]
            time.sleep(interval)
    except KeyboardInterrupt:
        print('Matrix multiplication interrupted by user.')


def main():
    args = parse_args()
    gpu_ids = [int(gpu_id) for gpu_id in args.gpus.split(',')]
    matrix_multiplication(gpu_ids, args.size, args.interval)

if __name__ == '__main__':
    main()