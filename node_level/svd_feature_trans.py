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

def x_svd(data, out_dim):
    assert data.shape[-1] >= out_dim
    U, S, _ = np.linalg.svd(data)
    newdata= np.matmul(U[:, :out_dim], np.diag(S[:out_dim]))
    return newdata


dataset_str = "Facebook"
data = loadmat('data/{}.mat'.format(dataset_str))

label = data['Label'] if ('Label' in data) else data['gnd']
attr = data['Attributes'] if ('Attributes' in data) else data['X']
adj = data['Network'] if ('Network' in data) else data['A']

feat = np.array(attr.todense())

print(feat)

import numpy as np
# from sklearn.decomposition import PCA
#
# pca = PCA(n_components=25)
# newX = pca.fit_transform(feat)  # 等价于pca.fit(X) pca.transform(X)
# invX = pca.inverse_transform(newX)  # 将降维后的数据转换成原始数据
# print(feat)
# print(newX)

newX = x_svd(feat, 8)
#
feat = csc_matrix(feat)
sio.savemat('data/{}_svd.mat'.format(dataset_str), \
            {'Network': adj, 'Label': label, 'Attributes': newX})
print('Done. The file is save as: dataset/{}_svd.mat \n'.format(dataset_str))
