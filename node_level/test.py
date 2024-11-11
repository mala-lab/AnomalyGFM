# -*- coding: utf-8 -*-
# @Author  : qiaohezhe
# @github : https://github.com/fengduqianhe
# @Date    :  8/19/2024 
# version： Python 3.7.8
# @File : test.py
# @Software: PyCharm
import torch

# 创建一个随机矩阵
matrix = torch.rand(3, 4)  # 3行4列的矩阵

# 计算每一行的和，保持原始维度用于广播
row_sums = matrix.sum(dim=1, keepdim=True)

# 将矩阵的每一行除以该行的和
normalized_matrix = matrix / row_sums

print("Original matrix:")
print(matrix)
print("\nMatrix after row-wise normalization:")
print(normalized_matrix)