# import torch 
# import time
# import os
# import argparse
# import shutil
# import sys
 
# def parse_args():
#     parser = argparse.ArgumentParser(description='Matrix multiplication')
#     parser.add_argument('--gpus', help='gpu amount', default='0', type=str)
#     parser.add_argument('--size', help='matrix size', required=True, type=int)
#     parser.add_argument('--interval', help='sleep interval', required=True, type=float)
#     args = parser.parse_args()
#     return args
 
 
# def matrix_multiplication(args):
 
#     a_list, b_list, result = [], [], []    
#     size = (args.size, args.size)
#     gpu_ids = [int(gpu_id) for gpu_id in args.gpus.split(',')]
    
#     for i in gpu_ids:
#         a_list.append(torch.rand(size, device=f'cuda:{i}'))
#         b_list.append(torch.rand(size, device=f'cuda:{i}'))
#         result.append(torch.rand(size, device=f'cuda:{i}'))
 
#     while True:
#         for i in range(len(gpu_ids)):
#             result[i] = a_list[i] * b_list[i]
#         time.sleep(args.interval)
 
# if __name__ == "__main__":
#     # usage:   python gpuhold.py  --size 38000 --gpus 4,5,6,7 --interval 0.01
#     args = parse_args()
#     matrix_multiplication(args)
    
    
    
    
from flask import Flask, request, render_template_string
import subprocess
import threading
import torch
import time

# 创建 Flask 应用
app = Flask(__name__)

# 您的矩阵乘法逻辑
def matrix_multiplication(size, gpu_ids, interval):
    a_list, b_list, result = [], [], []
    size = (size, size)
    gpu_ids = [int(gpu_id) for gpu_id in gpu_ids.split(',')]

    for i in gpu_ids:
        a_list.append(torch.rand(size, device=f'cuda:{i}'))
        b_list.append(torch.rand(size, device=f'cuda:{i}'))
        result.append(torch.rand(size, device=f'cuda:{i}'))

    while True:
        for i in range(len(gpu_ids)):
            result[i] = a_list[i] * b_list[i]
        time.sleep(interval)

# GPU 监控
def get_gpu_status():
    return subprocess.check_output(["watch", "-n", "1", "-c", "gpustat", "--color"]).decode()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        size = int(request.form['size'])
        gpus = request.form['gpus']
        interval = float(request.form['interval'])
        threading.Thread(target=matrix_multiplication, args=(size, gpus, interval)).start()
    gpu_status = get_gpu_status()
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <body>
        <h2>GPU Usage Control</h2>
        <form method="post">
            Matrix Size: <input type="text" name="size"><br>
            GPUs (e.g., 0,1,2): <input type="text" name="gpus"><br>
            Interval: <input type="text" name="interval"><br>
            <input type="submit" value="Submit">
        </form>
        <pre>{{ gpu_status }}</pre>
    </body>
    </html>
    """, gpu_status=gpu_status)
if __name__ == "__main__":
    app.run(port=5000)
    