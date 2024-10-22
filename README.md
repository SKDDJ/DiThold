# DiTHold
DiTHold is a Python script for performing matrix multiplication on GPU. It can be used to test the performance of GPUs.

## Usage
You can specify the number of GPUs, the size of the matrix, and the interval time between each operation through command line arguments. For example:
``` python dithold.py --size 38000 --gpus 4,5,6,7 --interval 0.01 ```

In this example, the script will perform matrix multiplication on GPUs 4, 5, 6, and 7, with a matrix size of 38000x38000, and an interval of 0.01 seconds between each operation.

## Parameters
--gpus: IDs of GPUs used for matrix multiplication, separated by commas. Default value is '0'.
--size: Size of the matrix. This parameter is required.
--interval: Interval time between each operation, in seconds. This parameter is required.
## Note
Make sure you have PyTorch installed on your machine and CUDA support. Additionally, the GPU IDs you specify must be valid, otherwise the script will not run properly.
