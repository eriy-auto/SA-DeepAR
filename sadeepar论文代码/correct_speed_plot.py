import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from mpl_toolkits.mplot3d import Axes3D


def inter_plot(plot_data, wt):  # plot_data - 数组格式
    x1 = [i for i in range(len(plot_data) - 1)]
    print(len(plot_data))
    ft_dict = {'family': 'Times New Roman', 'size': 20}
    plt.figure(figsize=(10, 6))
    plt.ylabel("power(KW)", fontdict=ft_dict)
    plt.plot(x1, plot_data[:-1, 13], lw=2, label="Test data", c="b")

    plt.plot(
        x1, plot_data[1:, 0], lw=2, c="C1", label="SADeepAR_Pre"
    )
    plt.fill_between(
        x1,
        plot_data[1:, 11],
        plot_data[1:, 12],
        color='forestgreen',
        alpha=0.2,
        label="95%intervals",
    )
    plt.fill_between(
        x1,
        plot_data[1:, 9],
        plot_data[1:, 10],
        color='limegreen',
        alpha=0.2,
        label="80%intervals",
    )
    plt.fill_between(
        x1,
        plot_data[1:, 7],
        plot_data[1:, 8],
        color='lightgreen',
        alpha=0.2,
        label="70%intervals",
    )
    plt.fill_between(
        x1,
        plot_data[1:, 5],
        plot_data[1:, 6],
        color='lightseagreen',
        alpha=0.2,
        label="60%intervals",
    )
    plt.fill_between(
        x1,
        plot_data[1:, 3],
        plot_data[1:, 4],
        color='plum',
        alpha=0.2,
        label="40%intervals",
    )
    plt.fill_between(
        x1,
        plot_data[1:, 1],
        plot_data[1:, 2],
        color='y',
        alpha=0.2,
        label="20%intervals",
    )

    plt.title(wt + ' Power pre', fontdict=ft_dict)
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1), prop={'family': 'Times New Roman', 'size': 18})
    plt.yticks(fontproperties='Times New Roman', size=16)
    plt.xticks(fontproperties='Times New Roman', size=16)
    plt.tight_layout()
    # plt.rc('font', family='Times New Roman', size=14)
    # plt.show()


def picp_cau(cau_data):  # cau_data - 数组格式
    sm_20, sm_40, sm_60, sm_70, sm_80, sm_95 = 0, 0, 0, 0, 0, 0
    m = len(cau_data)
    for i in range(m):
        if cau_data[i, 1] <= cau_data[i, -1] <= cau_data[i, 2]:
            sm_20 += 1
        if cau_data[i, 3] <= cau_data[i, -1] <= cau_data[i, 4]:
            sm_40 += 1
        if cau_data[i, 5] <= cau_data[i, -1] <= cau_data[i, 6]:
            sm_60 += 1
        if cau_data[i, 7] <= cau_data[i, -1] <= cau_data[i, 8]:
            sm_70 += 1
        if cau_data[i, 9] <= cau_data[i, -1] <= cau_data[i, 10]:
            sm_80 += 1
        if cau_data[i, 11] <= cau_data[i, -1] <= cau_data[i, 12]:
            sm_95 += 1
    picp_20 = sm_20 / m
    picp_40 = sm_40 / m
    picp_60 = sm_60 / m
    picp_70 = sm_70 / m
    picp_80 = sm_80 / m
    picp_95 = sm_95 / m
    print('picp_20', picp_20, 'picp_40', picp_40, 'picp_60', picp_60, 'picp_70', picp_70, 'picp_80'
          , picp_80, 'picp_95', picp_95)
    return picp_20, picp_40, picp_60, picp_70, picp_80, picp_95


def evaluate_cau(pre_data, target_data):
    n = len(pre_data)
    print('n', n)
    mean = np.mean(target_data)
    print('mean:', mean)
    rmse_sum = 0
    mae_sum = 0
    r2_sum_1 = 0
    r2_sum_2 = 0
    for i in range(n):
        rmse_va = (pre_data[i] - target_data[i]) ** 2
        rmse_sum += rmse_va
        mae_va = abs(pre_data[i] - target_data[i])
        mae_sum += mae_va
        r2_va_1 = (pre_data[i] - target_data[i])**2
        r2_va_2 = (target_data[i] - mean) ** 2
        r2_sum_1 += r2_va_1
        r2_sum_2 += r2_va_2
    rmse = np.sqrt(rmse_sum/n)
    rmse_2 = np.sqrt(mean_squared_error(target_data, pre_data))
    mae = mae_sum/n
    mae_2 = mean_absolute_error(target_data, pre_data)
    r2 = 1 - r2_sum_1/r2_sum_2
    r2_2 = r2_score(target_data, pre_data)
    # rmse = np.sqrt(np.sum((pre_data - target_data)**2)/n)
    # mape = np.sum(np.abs((pre_data - target_data)/target_data))/n
    # r2 = 1 - np.sum((pre_data - target_data)**2)/np.sum((pre_data - mean)**2)
    print('rmse:', rmse)
    print('rmse_2:', rmse_2)
    print('mae:', mae)
    print('mae_2:', mae_2)
    print('r2:', r2)
    print('r2_2:', r2_2)
    # return rmse, mape, r2


def power_pdf(sadp_data, enbpi_data, sarima_data):
    te_data_power = sum([i / 6 for i in sadp_data[3151:3501, -1]])
    sadp_data_power = sum([i / 6 for i in sadp_data[3152:3502, 0]])

    ep_data_power = sum([i / 6 for i in enbpi_data[3150:3500, 0]])

    sari_data_power = sum([(i + 200) / 6 for i in sarima_data[3137:3487, 0]])

    print('te_data_power:', te_data_power)
    print('sadp_data_power:', sadp_data_power)
    print('ep_data_power:', ep_data_power)

    # 生成正态分布随机数
    sadp_mu = sadp_data_power
    sadp_sigma = np.sqrt(sum([((sadp_data[i, 1] - sadp_data[i, 0]) / 18) ** 2 for i in range(3152, 3502)]))
    sadp_samples_normal = np.random.normal(sadp_mu, sadp_sigma, 100000)

    enbpi_mu = ep_data_power
    enbpi_sigma = np.sqrt(sum([((enbpi_data[i, 1] - enbpi_data[i, 0]) / 18) ** 2 for i in range(3150, 3500)]))
    enbpi_samples_normal = np.random.normal(enbpi_mu, enbpi_sigma, 100000)

    sari_mu = sari_data_power
    sari_sigma = np.sqrt(sum([((sarima_data[i, 1] - sarima_data[i, 0]) / 18) ** 2 for i in range(3137, 3487)]))
    sari_samples_normal = np.random.normal(sari_mu, sari_sigma, 100000)

    # 绘制正态分布的概率密度函数图形
    plt.figure(figsize=(10, 6))
    n, sadp_bins, patches = plt.hist(sadp_samples_normal, bins=100, density=True, alpha=0.5, label='sadeepar')
    n, enbpi_bins, patches = plt.hist(enbpi_samples_normal, bins=100, density=True, alpha=0.5,
                                      label='enbpi')
    n, sarima_bins, patches = plt.hist(sari_samples_normal, bins=100, density=True, alpha=0.5,
                                       label='sarima')
    # 直方图函数，x为x轴的值，normed=1表示为概率密度，即和为一，绿色方块，色深参数0.5.返回n个概率，直方块左边线的x值，及各个方块对象

    sadp_y = mlab.normpdf(sadp_bins, sadp_mu, sadp_sigma)  # 拟合一条最佳正态分布曲线y
    enbpi_y = mlab.normpdf(enbpi_bins, enbpi_mu, enbpi_sigma)  # 拟合一条最佳正态分布曲线y
    sarima_y = mlab.normpdf(sarima_bins, sari_mu, sari_sigma)  # 拟合一条最佳正态分布曲线y
    max_dp, max_en, max_sa = max(sadp_y), max(enbpi_y), max(sarima_y)

    plt.plot(sadp_bins, sadp_y, color='red', label='sadeepar_PDF')
    plt.plot(enbpi_bins, enbpi_y, color='y', label='enbpi_PDF')
    plt.plot(sarima_bins, sarima_y, color='c', label='sarima_PDF')
    plt.vlines(te_data_power, 0, max([max_sa, max_en, max_dp]), colors="c", linestyles="dashed", label='True_data')
    plt.xlabel('Power(KW·H)', fontdict={'family': 'Times New Roman', 'size': 20})
    plt.ylabel('Probability Density', fontdict={'family': 'Times New Roman', 'size': 20})
    plt.title('Generated Power Normal Distribution', fontdict={'family': 'Times New Roman', 'size': 20})
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1), prop={'family': 'Times New Roman', 'size': 18})
    plt.tight_layout()
    plt.yticks(fontproperties='Times New Roman', size=16)
    plt.xticks(fontproperties='Times New Roman', size=16)
    # plt.rc('font', family='Times New Roman', size=16)


def sadp_pdf_plot(sadp_data):
    te_data_power = sum([i / 6 for i in sadp_data[3151:3501, -1]])
    sadp_data_power = sum([i / 6 for i in sadp_data[3152:3502, 0]])

    # 生成正态分布随机数
    sadp_mu = sadp_data_power
    sadp_sigma_20 = np.sqrt(sum([((sadp_data[i, 2] - sadp_data[i, 0]) / 18) ** 2 for i in range(3151, 3501)]))
    sadp_sigma_40 = np.sqrt(sum([((sadp_data[i, 4] - sadp_data[i, 0]) / 18) ** 2 for i in range(3151, 3501)]))
    sadp_sigma_60 = np.sqrt(sum([((sadp_data[i, 6] - sadp_data[i, 0]) / 18) ** 2 for i in range(3151, 3501)]))
    sadp_sigma_70 = np.sqrt(sum([((sadp_data[i, 8] - sadp_data[i, 0]) / 18) ** 2 for i in range(3151, 3501)]))
    sadp_sigma_80 = np.sqrt(sum([((sadp_data[i, 10] - sadp_data[i, 0]) / 18) ** 2 for i in range(3151, 3501)]))
    sadp_sigma_95 = np.sqrt(sum([((sadp_data[i, 12] - sadp_data[i, 0]) / 18) ** 2 for i in range(3151, 3501)]))
    sadp_samples_normal_20 = np.random.normal(sadp_mu, sadp_sigma_20, 100000)
    sadp_samples_normal_40 = np.random.normal(sadp_mu, sadp_sigma_40, 100000)
    sadp_samples_normal_60 = np.random.normal(sadp_mu, sadp_sigma_60, 100000)
    sadp_samples_normal_70 = np.random.normal(sadp_mu, sadp_sigma_70, 100000)
    sadp_samples_normal_80 = np.random.normal(sadp_mu, sadp_sigma_80, 100000)
    sadp_samples_normal_95 = np.random.normal(sadp_mu, sadp_sigma_95, 100000)

    # 绘制正态分布的概率密度函数图形
    plt.figure(figsize=(10, 6))
    n, sadp_bins_20, patches = plt.hist(sadp_samples_normal_20, bins=100, density=True, alpha=0.5, label='20%intervals')
    n, sadp_bins_40, patches = plt.hist(sadp_samples_normal_40, bins=100, density=True, alpha=0.5, label='40%intervals')
    n, sadp_bins_60, patches = plt.hist(sadp_samples_normal_60, bins=100, density=True, alpha=0.5, label='60%intervals')
    n, sadp_bins_70, patches = plt.hist(sadp_samples_normal_70, bins=100, density=True, alpha=0.5, label='70%intervals')
    n, sadp_bins_80, patches = plt.hist(sadp_samples_normal_80, bins=100, density=True, alpha=0.5, label='80%intervals')
    n, sadp_bins_95, patches = plt.hist(sadp_samples_normal_95, bins=100, density=True, alpha=0.5, label='95%intervals')
    # 直方图函数，x为x轴的值，normed=1表示为概率密度，即和为一，绿色方块，色深参数0.5.返回n个概率，直方块左边线的x值，及各个方块对象

    sadp_y_20 = mlab.normpdf(sadp_bins_20, sadp_mu, sadp_sigma_20)  # 拟合一条最佳正态分布曲线y
    sadp_y_40 = mlab.normpdf(sadp_bins_40, sadp_mu, sadp_sigma_40)
    sadp_y_60 = mlab.normpdf(sadp_bins_60, sadp_mu, sadp_sigma_60)
    sadp_y_70 = mlab.normpdf(sadp_bins_70, sadp_mu, sadp_sigma_70)
    sadp_y_80 = mlab.normpdf(sadp_bins_80, sadp_mu, sadp_sigma_80)
    sadp_y_95 = mlab.normpdf(sadp_bins_95, sadp_mu, sadp_sigma_95)

    # 拟合一条最佳正态分布曲线y
    plt.plot(sadp_bins_20, sadp_y_20, color='C1', label='20%PDF')
    plt.plot(sadp_bins_40, sadp_y_40, color='C2', label='40%PDF')
    plt.plot(sadp_bins_60, sadp_y_60, color='C3', label='60%PDF')
    plt.plot(sadp_bins_70, sadp_y_70, color='C4', label='70%PDF')
    plt.plot(sadp_bins_80, sadp_y_80, color='C5', label='80%PDF')
    plt.plot(sadp_bins_95, sadp_y_95, color='C6', label='95%PDF')

    plt.vlines(te_data_power, 0, max(sadp_y_20), colors="c", linestyles="dashed", label='True_data')
    plt.xlabel('Power(KW·H)', fontdict={'family': 'Times New Roman', 'size': 20})
    plt.ylabel('Probability Density', fontdict={'family': 'Times New Roman', 'size': 20})
    plt.title('Generated Power Normal Distribution', fontdict={'family': 'Times New Roman', 'size': 20})
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1), prop={'family': 'Times New Roman', 'size': 18})
    plt.tight_layout()
    plt.yticks(fontproperties='Times New Roman', size=16)
    plt.xticks(fontproperties='Times New Roman', size=16)


def point_compare_plot(bpnn_data, lstm_data, deepar_data, sadp_data, real_data):
    plt.figure(figsize=(10, 6))
    plt.plot(bpnn_data[1:], color='red', label='bpnn_pre')
    plt.plot(lstm_data[1:], color='y', label='lstm_pre')
    plt.plot(deepar_data[1:, 0], color='c', label='deepar_pre')
    plt.plot(sadp_data[1:, 0], color='C2', label='sa-deepar_pre')
    plt.plot(real_data[:-1], color='C4', label='real power')
    plt.ylabel('Power(KW)', fontdict={'family': 'Times New Roman', 'size': 20})
    plt.xlabel('Time(10min)', fontdict={'family': 'Times New Roman', 'size': 20})
    # plt.ylim([700, 2500])
    plt.title('Comparison of power prediction results(3-steps)', fontdict={'family': 'Times New Roman', 'size': 20})
    plt.legend(loc='upper left',
               prop={'family': 'Times New Roman', 'size': 16}, shadow=False)  # 'upper right' 'best'
    plt.tight_layout()
    plt.yticks(fontproperties='Times New Roman', size=16)
    plt.xticks(fontproperties='Times New Roman', size=16)


def wt_compare_plot(bpnn_data, lstm_data, deepar_data, sadp_data, real_data):
    x1 = [i for i in range(len(real_data) - 1)]
    plt.figure(figsize=(10, 6))
    plt.plot(bpnn_data[1:], color='red', label='bpnn_pre')
    plt.plot(lstm_data[1:], color='y', label='lstm_pre')
    plt.plot(deepar_data[1:, 0], color='c', label='deepar_pre')
    plt.plot(sadp_data[1:, 0], color='C2', label='sa-deepar_pre')
    plt.plot(real_data[:-1], color='C4', label='real power')
    plt.fill_between(
        x1,
        sadp_data[1:, 3],
        sadp_data[1:, 4],
        color='C2',
        alpha=0.2,
        label="40%intervals",
    )
    plt.ylabel('Power(KW)', fontdict={'family': 'Times New Roman', 'size': 20})
    plt.xlabel('Time(10min)', fontdict={'family': 'Times New Roman', 'size': 20})
    # plt.ylim([750, 2500])
    plt.title('Comparison of T05 power prediction results', fontdict={'family': 'Times New Roman', 'size': 20})
    plt.legend(loc='upper left', bbox_to_anchor=(0, 1),
               prop={'family': 'Times New Roman', 'size': 16}, shadow=False)  # 'upper right' 'best'
    plt.tight_layout()
    plt.yticks(fontproperties='Times New Roman', size=16)
    plt.xticks(fontproperties='Times New Roman', size=16)


def wind_speed_plot(scada_wind, light_wind):
    plt.figure(figsize=(10, 6))
    plt.plot(scada_wind, color='g', label='V$_1$')  # SCADA wind
    plt.plot(light_wind, color='b', label='V$_0$')  # Light wind
    plt.ylabel('Wind speed(m/s)', fontdict={'family': 'Times New Roman', 'size': 20})
    plt.xlabel('Time(s)', fontdict={'family': 'Times New Roman', 'size': 20})
    plt.title('Comparison of wind speed', fontdict={'family': 'Times New Roman', 'size': 20})
    plt.legend(loc='upper right',  # bbox_to_anchor=(0, 1),
               prop={'family': 'Times New Roman', 'size': 16}, shadow=False)  # 'upper right' 'best'
    plt.tight_layout()
    plt.yticks(fontproperties='Times New Roman', size=16)
    plt.xticks(fontproperties='Times New Roman', size=16)


# 读取数据文件
# 概率预测结果比较
'''
file_name_1 = './pre_data/30-T02-sadeepar-功率概率预测结果-20-95.csv'
file_name_2 = './pre_data/30-T04-sadeepar-功率概率预测结果-20-95.csv'
file_name_3 = './pre_data/30-T05-sadeepar-功率概率预测结果-20-95.csv'
file_name_4 = './pre_data/11-T02-enbpi-修正风速功率概率预测结果-1步-2.csv'
file_name_5 = './pre_data/11-T02-sarima-修正风速功率概率预测结果-1步-2.csv'


df_t02 = pd.read_csv(file_name_1, encoding='gbk')
t02_data = df_t02.to_numpy()
index_02 = np.where(t02_data[:, -1] == 0)
# t02_data = np.delete(t02_data, index_02, axis=0)

df_t04 = pd.read_csv(file_name_2, encoding='gbk')
t04_data = df_t04.to_numpy()
index_04 = np.where(t04_data[:, -1] == 0)
t04_data = np.delete(t04_data, index_04, axis=0)

df_t05 = pd.read_csv(file_name_3, encoding='gbk')
t05_data = df_t05.to_numpy()
index_05 = np.where(t05_data[:, -1] == 0)
t05_data = np.delete(t05_data, index_05, axis=0)

df_enbpi = pd.read_csv(file_name_4)
ebp_data = df_enbpi.to_numpy()
# ebp_data = np.delete(ebp_data, index_02, axis=0)

df_sarima = pd.read_csv(file_name_5)
sar_data = df_sarima.to_numpy()
# sar_data = np.delete(sar_data, index_02, axis=0)
'''

'''
picp_cau(t02_data[1500:1700, :])
picp_cau(t04_data[1500:1700, :])
picp_cau(t05_data[1500:1700, :])
inter_plot(t02_data[1500:1700, :], 'T02')
inter_plot(t04_data[1500:1700, :], 'T04')
inter_plot(t05_data[1500:1700, :], 'T05')
'''

# 不同风机算法对比数据
'''
file_name_1 = './概率预测结果/结果数据/BPNN预测结果.csv'  # BPNN预测结果  2步  3步
file_name_2 = './概率预测结果/结果数据/lstm预测结果.csv'  # lstm预测结果  2步  3步
file_name_3 = './概率预测结果/结果数据/T01-2014功率概率预测结果(deepar).csv'  # T01-2014功率概率预测结果(deepar)  deepar功率概率预测结果-2步  3步
file_name_4 = './概率预测结果/结果数据/T01-2014功率概率预测结果-4.csv'  # T01-2014功率概率预测结果-4 T01-2014功率概率预测结果-2步  3步
file_name_5 = './概率预测结果/结果数据/T01-2014功率实际值.csv'
file_name_6 = './pre_data/28-T05-sadeepar-功率概率预测结果-20-40.csv'
'''
'''
file_name_1 = './概率预测结果/结果数据/BPNN预测结果-2步.csv'  # BPNN预测结果  2步  3步
file_name_2 = './概率预测结果/结果数据/lstm预测结果-2步.csv'  # lstm预测结果  2步  3步
file_name_3 = './概率预测结果/结果数据/deepar功率概率预测结果-2步.csv'  # T01-2014功率概率预测结果(deepar)  deepar功率概率预测结果-2步  3步
file_name_4 = './概率预测结果/结果数据/T01-2014功率概率预测结果-2步.csv'  # T01-2014功率概率预测结果-4 T01-2014功率概率预测结果-2步  3步
file_name_5 = './概率预测结果/结果数据/T01-2014功率实际值.csv'
file_name_6 = './pre_data/28-T05-sadeepar-功率概率预测结果-20-40.csv'
'''

'''
file_name_1 = './概率预测结果/结果数据/BPNN预测结果-3步.csv'  # BPNN预测结果  2步  3步
file_name_2 = './概率预测结果/结果数据/lstm预测结果-3步.csv'  # lstm预测结果  2步  3步
file_name_3 = './概率预测结果/结果数据/deepar功率概率预测结果-3步.csv'  # T01-2014功率概率预测结果(deepar)  deepar功率概率预测结果-2步  3步
file_name_4 = './概率预测结果/结果数据/T01-2014功率概率预测结果-3步.csv'  # T01-2014功率概率预测结果-4 T01-2014功率概率预测结果-2步  3步
file_name_5 = './概率预测结果/结果数据/T01-2014功率实际值.csv'
file_name_6 = './pre_data/28-T05-sadeepar-功率概率预测结果-20-40.csv'


df_real = pd.read_csv(file_name_5, encoding='utf-8')
real_data = df_real.to_numpy()
index_0 = np.where(real_data[:, 0] == 0)
real_data = np.delete(real_data, index_0, axis=0)

df_bpnn = pd.read_csv(file_name_1, encoding='utf-8')
bpnn_data = df_bpnn.to_numpy()
bpnn_data = np.delete(bpnn_data, index_0, axis=0)

df_lstm = pd.read_csv(file_name_2, encoding='utf-8')
lstm_data = df_lstm.to_numpy()
lstm_data = np.delete(lstm_data, index_0, axis=0)

df_deepar = pd.read_csv(file_name_3, encoding='utf-8')
deepar_data = df_deepar.to_numpy()
deepar_data = np.delete(deepar_data, index_0, axis=0)

df_sadp = pd.read_csv(file_name_4, encoding='utf-8')
sadp_data = df_sadp.to_numpy()
sadp_data = np.delete(sadp_data, index_0, axis=0)

df_sadp_2 = pd.read_csv(file_name_6, encoding='utf-8')
sadp_data_2 = df_sadp_2.to_numpy()
sadp_data_2 = np.delete(sadp_data_2, index_0, axis=0)

point_compare_plot(bpnn_data=bpnn_data[400:600, :], lstm_data=lstm_data[400:600, :], deepar_data=deepar_data[400:600, :],
                   sadp_data=sadp_data[400:600, :], real_data=real_data[401:601, :])  # 2000:400 5000:700

# wt_compare_plot(bpnn_data=bpnn_data[200:350, :], lstm_data=lstm_data[200:350, :], deepar_data=deepar_data[200:350, :],
#                 sadp_data=sadp_data_2[200:350, :], real_data=real_data[200:350, :])  # 200:350
evaluate_cau(pre_data=bpnn_data[400:600, :], target_data=real_data[403:603, :])
evaluate_cau(pre_data=lstm_data[400:600, :], target_data=real_data[403:603, :])
evaluate_cau(pre_data=deepar_data[400:600, 0].reshape(-1, 1), target_data=real_data[403:603, :])
evaluate_cau(pre_data=sadp_data[400:600, 0].reshape(-1, 1), target_data=real_data[403:603, :])
# evaluate_cau(pre_data=sadp_data_2[5000:700, 0].reshape(-1, 1), target_data=real_data[5000:700, :])

# power_pdf(t02_data, ebp_data, sar_data)
# sadp_pdf_plot(t02_data)
plt.show()
'''

#  风速比较数据
wd_file_1 = './概率预测结果/结果数据/激光雷达数据1_201909.xlsx'
wd_file_2 = './概率预测结果/结果数据/风速修正比较-scada风速BPNN预测结果-1.csv'
wd_file_3 = './概率预测结果/结果数据/风速修正比较-激光雷达风速BPNN预测结果-1.csv'
wd_file_4 = './概率预测结果/结果数据/风速修正比较-修正风速BPNN预测结果-1.csv'
wd_file_5 = './概率预测结果/结果数据/预测用数据.csv'
df_real = pd.read_excel(wd_file_1)
wind_data = df_real.to_numpy()
scada_data = wind_data[:, 1]
light_data = wind_data[:, 7]

wind_speed_plot(scada_wind=scada_data[:1000], light_wind=light_data[:1000])

df_scada = pd.read_csv(wd_file_2)
scada_pre = df_scada.to_numpy()
df_light = pd.read_csv(wd_file_3)
light_pre = df_light.to_numpy()
df_fxi = pd.read_csv(wd_file_4)
fxi_pre = df_fxi.to_numpy()
power_d = pd.read_csv(wd_file_5)
power_data = power_d.to_numpy()
power_data = power_data[:798, 3]
evaluate_cau(pre_data=scada_pre[:], target_data=power_data[:])  #400:600
evaluate_cau(pre_data=light_pre[:], target_data=power_data[:])
evaluate_cau(pre_data=fxi_pre[:], target_data=power_data[:])
plt.figure(figsize=(10, 6))
plt.plot(scada_pre[:]+500, color='g', label='Scada pre')
plt.plot(light_pre[:]+500, color='b', label='Light pre')
plt.plot(fxi_pre[:]+500, color='y', label='Proposed model pre')
plt.plot(power_data[:]+500, color='k', label='Real')
plt.ylabel('Wind Power(KW)', fontdict={'family': 'Times New Roman', 'size': 20})
plt.xlabel('Time', fontdict={'family': 'Times New Roman', 'size': 20})
plt.title('Comparison of power prediction results', fontdict={'family': 'Times New Roman', 'size': 20})
plt.legend(loc='lower right',  # bbox_to_anchor=(0, 1),
               prop={'family': 'Times New Roman', 'size': 16}, shadow=False)  # 'upper right' 'best'
plt.tight_layout()
plt.yticks(fontproperties='Times New Roman', size=16)
plt.xticks(fontproperties='Times New Roman', size=16)
plt.show()



