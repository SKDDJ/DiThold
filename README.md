# Train Dit

`train_dit` is a Python package designed to occupy GPU resources by performing matrix multiplication on specified GPUs to keep them active.

## Usage

You can specify the number of GPUs, the size of the matrix, and the interval time between each operation through command line arguments. For example:

```bash
python train_dit.py --size 38000 --gpus 4,5,6,7 --interval 0.01
```

In this example, the script will perform matrix multiplication on GPUs 4, 5, 6, and 7, with a matrix size of 38000x38000, and an interval of 0.01 seconds between each operation.

## Parameters

- `--gpus`: IDs of GPUs used for matrix multiplication, separated by commas. Default value is `'0,1,2,3,4,5,6,7'`.
- `--size`: Size of the matrix. This parameter is required.
- `--interval`: Interval time between each operation, in seconds. This parameter is required.

## Note

Make sure you have PyTorch installed on your machine and CUDA support. Additionally, the GPU IDs you specify must be valid; otherwise, the script will not run properly.

**Author**

[Yiming Shi](https://academic.shiym.top)

**Contact**: [yimingshi666@gmail.com](mailto:yimingshi666@gmail.com)

**GitHub**: [https://github.com/SKDDJ/train_dit](https://github.com/SKDDJ/train_dit)