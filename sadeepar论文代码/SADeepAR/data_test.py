from tensorflow.python.framework.ops import disable_eager_execution
from numpy.random import normal
import tqdm
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from keras import backend as K
import time


def createXY_all(xdata, ydata, n_past, pre_le):
    dataX = []
    dataY = []
    print('dataset', xdata.shape[0])
    for i in range(n_past, xdata.shape[0] - pre_le, pre_le):
        dataX.append(xdata[i - n_past:i])
        dataY.append(ydata[i:i + pre_le])
    return np.array(dataX), np.array(dataY)


def cau_mean(data, time_step):
    # le = len(data.shape[0])
    data_mean = []
    for i in range(int(data.shape[0] / time_step)):
        data_mean.append(np.mean(data[i * time_step:(i + 1) * time_step], axis=0))
    return np.array(data_mean)


start = time.time()
filepath = '预测集合数据.csv'  # 文件路径 [激光风速， SCADA风速， 修正风速， 功率];
# ['激光雷达风速', 'SCADA风速', '风向', '叶片角度', '转速', '修正风速', '功率']
df_data = pd.read_csv(filepath)
data = df_data.to_numpy()
data = np.array(data)
# X_data = data[:50000, :4]
Y_data = data[:66000, 6].reshape(-1, 1)
X_data = data[:66000, 2:6]
X_data = X_data.reshape(X_data.shape[0], X_data.shape[1])
# print(data.shape)
print(X_data[:5])

t_ydata = cau_mean(Y_data, 1).reshape(-1, 1)
print('t_ydata', t_ydata.shape)
t_xdata = cau_mean(X_data, 1)
print('t_xdata', t_xdata.shape)
'''
plt.figure()
plt.subplot(511)
plt.plot(t_ydata)
plt.subplot(512)
plt.plot(t_xdata[:, 0])
plt.subplot(513)
plt.plot(t_xdata[:, 1])
plt.subplot(514)
plt.plot(t_xdata[:, 2])
plt.subplot(515)
plt.plot(t_xdata[:, 3])
plt.show()'''