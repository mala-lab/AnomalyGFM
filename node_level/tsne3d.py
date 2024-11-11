import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D


# 加载数据
# def get_data():
# 	"""
# 	:return: 数据集、标签、样本数量、特征数量
# 	"""
# 	digits = datasets.load_digits(n_class=10)
# 	data = digits.data		# 图片特征
# 	label = digits.target		# 图片标签
# 	n_samples, n_features = data.shape		# 数据集的形状
# 	return data, label, n_samples, n_features

# def get_data():
# 	data = sio.loadmat('./embedding/Amazon_no_isolate_net_upu_1_00_raw.mat')
# 	label = np.squeeze(data['Label'])
# 	label =label.astype(int)
# 	data = np.squeeze(data['emb'])
#
# 	return data, label

# 对样本进行预处理并画图
def plot_embedding(data, label):
    """
	:param data:数据集
	:param label:样本标签
	:param title:图像标题
	:return:图像
	"""
    # yellow green red
    colors = np.array(['r', 'g', 'y'])
    # marker = np.array([".", "^", "^"])
    label_name = np.array([ 'Abnormal', 'Normal'])
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)  # 对数据进行归一化处理
    fig = plt.figure()  # 创建图形实例
    # ax = plt.subplot(111)  # 创建子图
    ax = Axes3D(fig)
    normal_index = np.array([index for (index, value) in enumerate(label) if value == 0])
    abnormal_index = np.array([index for (index, value) in enumerate(label) if value == 1])

    type1 = ax.scatter(data[normal_index, 0], data[normal_index, 1], data[normal_index, 2],
                       color='r', marker='x', alpha=1 / 3)
    type2 = ax.scatter(data[abnormal_index, 0], data[abnormal_index, 1], data[abnormal_index, 2],
                       color='g', marker='o', alpha=1 / 3,  s=100)


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend((type1, type2), ("Normal",  "Abnormal",), ncol=1,  markerscale=3, fontsize=23, loc=2)
    plt.yticks()

    # plt.title(title, fontsize=14)
    # 返回值
    return fig



# main
if __name__ == '__main__':
    dataset = 'reddit'
    data = sio.loadmat('tsne/{}_svd_1170.mat'.format(dataset))
    emb_residual_test = data['emb_residual_test']
    emb_residual_train = data['emb_residual_train']

    label_train = np.squeeze(data['label_train'])
    label_test = np.squeeze(data['label_test'])

    emb_residual_test_normal = emb_residual_test[label_test == 0]

    emb_residual_train_normal = emb_residual_train[label_train == 0]

    emb_residual_test_abnormal = emb_residual_test[label_test == 1][:8000]

    emb_residual_train_abnormal = emb_residual_train[label_train == 1]

    emb_residual = np.concatenate((emb_residual_test_abnormal, emb_residual_train_abnormal), 0)
    label = np.concatenate((np.zeros(emb_residual_test_abnormal.shape[0]), np.ones(emb_residual_train_abnormal.shape[0])))

    ts = TSNE(n_components=3, init='pca', random_state=2)
    emb_selected = ts.fit_transform(emb_residual)
    fig = plot_embedding(np.array(emb_selected), label)
    plt.show()