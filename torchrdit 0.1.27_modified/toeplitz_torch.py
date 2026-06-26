# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 10:21:08 2025

@author: dell
"""
import torch

def toeplitz_torch(first_row, first_col):
      n = len(first_row)
      m = len(first_col)
      
      dtype=first_row.dtype

# 初始化Toeplitz矩阵
      #toeplitz_matrix = torch.zeros((n, m), dtype=torch.float32)
      toeplitz_matrix = torch.zeros((n, m), dtype=dtype) #torch.complex128

# 填充Toeplitz矩阵
      for i in range(n):
       for j in range(m):
        if i < j:
            toeplitz_matrix[i, j] = first_col[j - i]
        else:
            toeplitz_matrix[i, j] = first_row[i - j] if i >= j else first_row[0]

      return toeplitz_matrix