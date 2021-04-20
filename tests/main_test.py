import unittest

from ngpuinfo import NGPUInfo


class MainTest(unittest.TestCase):

    def test_gpu_number(self):
        Gpus = NGPUInfo.list_gpus()
        self.assertEqual(len(Gpus), NGPUInfo.NUMBERS)

    def test_gpu_info(self):
        Gpus = NGPUInfo.list_gpus()
        print(Gpus[0].id)
        print(Gpus[0].name)
        print(Gpus[0].mem_info())
