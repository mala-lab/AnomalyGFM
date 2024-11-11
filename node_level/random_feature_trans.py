# -*- coding: utf-8 -*-
# @Author  : qiaohezhe
# @github : https://github.com/fengduqianhe
# @Date    :  7/2/2024 
# version： Python 3.7.8
# @File : svd_feature_trans.py
# @Software: PyCharm
from scipy.linalg import svd
import matplotlib.pyplot as plt
import numpy as np
import numpy
import scipy.sparse as sp
from scipy.io import loadmat
import scipy.io as sio
from scipy.sparse import csc_matrix


def random_projection(data, target_dim):
    """
        Randomly project the data to reduce dimensionality.

        parameter:
        data – Input data, a two-dimensional numpy array of shape (m, n), where m is the number of samples and n is the number of features.
        target_dim -- the projected data dimension.

        return:
        projected_data – Projected data, a two-dimensional numpy array of shape (m, target_dim).
    """
    # Dimensions of original data
    original_dim = data.shape[1]

    # Generate a random matrix with shape (target_dim, original_dim)
    # Each element of the random matrix is drawn from the standard normal distribution
    random_matrix = np.random.randn(original_dim, target_dim)

    # Use a random matrix to project the original data
    projected_data = np.dot(data, random_matrix)

    return projected_data


dataset_str = "Amazon"
data = loadmat('src/data/{}.mat'.format(dataset_str))

label = data['Label'] if ('Label' in data) else data['gnd']
attr = data['Attributes'] if ('Attributes' in data) else data['X']
adj = data['Network'] if ('Network' in data) else data['A']

feat = np.array(attr.todense())
print(feat)

target_dim = 300
newX = random_projection(feat, target_dim)
print(newX.shape)


# newX = csc_matrix(newX)
# sio.savemat('src/data/{}_random.mat'.format(dataset_str), \
#             {'Network': adj, 'Label': label, 'Attributes': newX})
# print('Done. The file is save as: dataset/{}_random.mat \n'.format(dataset_str))
