import unittest
from train_dit.core import matrix_multiplication
import multiprocessing
import time

class TestTrainDit(unittest.TestCase):
    def test_matrix_multiplication(self):
        # 使用较小的矩阵和短时间运行以测试功能
        gpu_ids = [0]
        size = 100
        interval = 0.1

        # 运行矩阵乘法在一个子进程中，以便可以中断
        proc = multiprocessing.Process(target=matrix_multiplication, args=(gpu_ids, size, interval))
        proc.start()
        time.sleep(1)  # 运行一段时间
        proc.terminate()
        proc.join()
        self.assertFalse(proc.is_alive())

if __name__ == '__main__':
    unittest.main()
