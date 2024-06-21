import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.mlab as mlab

file_name_1 = './pre_data/11-T02-sadeepar-修正风速功率概率预测结果-1步.csv'
file_name_2 = './pre_data/11-T02-enbpi-修正风速功率概率预测结果-1步-2.csv'
file_name_3 = './pre_data/11-T02-sarima-修正风速功率概率预测结果-1步-2.csv'
df_sadp = pd.read_csv(file_name_1)
sadp_data = df_sadp.to_numpy()

df_enbpi = pd.read_csv(file_name_2)
enbpi_data = df_enbpi.to_numpy()

df_sarima = pd.read_csv(file_name_3)
sarima_data = df_sarima.to_numpy()

# print('te_data_power:', te_data_power)


x1 = [i for i in range(len(sadp_data[3151:3501, 5]))]
plt.figure(figsize=(12, 7))

plt.ylabel("power(KW)", fontdict={'family': 'Times New Roman', 'size': 20})
plt.plot(x1, sadp_data[3151:3501, 5], lw=2, label="Test data", c="b")

plt.plot(
    x1, sadp_data[3152:3502, 0], lw=2, c="C1", label="sadeepar_Pre"
)
plt.plot(
    x1, enbpi_data[3150:3500, 0], lw=2, c="C2", label="enbpi_Pre"
)
plt.plot(
             x1, sarima_data[3137:3487, 0]+200, lw=2, c="c", label="sarima_Pre"
         )
plt.fill_between(
    x1,
    sadp_data[3152:3502, 1],
    sadp_data[3152:3502, 2],
    color="C1",
    alpha=0.2,
    label="sadeepar_Pre intervals",
)
plt.fill_between(
    x1,
    enbpi_data[3150:3500, 1],
    enbpi_data[3150:3500, 2],
    color="C2",
    alpha=0.2,
    label="enbpi_Pre intervals",
)

plt.fill_between(
            x1,
            sarima_data[3137:3487, 1]+200,
            sarima_data[3137:3487, 2]+200,
            color="c",
            alpha=0.2,
            label="sarima_Pre intervals",
        )

plt.title('Power Predict Results', fontdict={'family': 'Times New Roman', 'size': 20})
plt.legend(loc='upper right', bbox_to_anchor=(1, 1), prop={'family': 'Times New Roman', 'size': 18})
plt.yticks(fontproperties='Times New Roman', size=16)
plt.xticks(fontproperties='Times New Roman', size=16)
plt.xlim([0, 350])
plt.tight_layout()
plt.show()

#  计算指标

n = len(sadp_data[1:, 5])  # 3150-3500
m = len(sadp_data[3151:3501, 5])
sa_dp_rmse = np.sqrt((np.sum((sadp_data[3151:3501, 5] - sadp_data[3152:3502, 0]) ** 2) / n))
sa_dp_mape = np.sum(np.abs(sadp_data[3151:3501, 5] - sadp_data[3152:3502, 0])) / n
sa_dp_picp = 0
sm = 0
for i in range(3152, 3152+m):

    if sadp_data[i, 1] <= sadp_data[i-1, 5] <= sadp_data[i, 2]:
        sm += 1
sa_dp_picp = sm/m
pinaw = np.sum(sadp_data[3152:3502, 2] - sadp_data[3152:3502, 1])
R = np.max(sadp_data[3151:3501, 5]) - np.min(sadp_data[3151:3501, 5])
sa_dp_pinaw = pinaw/(R*m)
print('sa_dp_picp:', sa_dp_picp, 'sa_dp_pinaw:', sa_dp_pinaw)
print('sadp_rmse:', sa_dp_rmse)
print('sadp_mae:', sa_dp_mape)

enbpi_rmse = np.sqrt((np.sum((sadp_data[3151:3501, 5] - enbpi_data[3150:3500, 0]) ** 2) / n))
enbpi_mape = np.sum(np.abs(sadp_data[3151:3501, 5] - enbpi_data[3150:3500, 0])) / n
enbpi_picp = 0
sm = 0
for i in range(3150, 3150+m):

    if enbpi_data[i, 1] <= sadp_data[i+1, 5] <= enbpi_data[i, 2]:
        sm += 1
enbpi_picp = sm/m
pinaw = np.sum(enbpi_data[3150:3500, 2] - enbpi_data[3150:3500, 1])
enbpi_pinaw = pinaw/(R*m)
print('enbpi_picp:', enbpi_picp, 'enbpi_pinaw:', enbpi_pinaw)
print('enbpi_rmse:', enbpi_rmse)
print('enbpi_mae:', enbpi_mape)

sarima_rmse = np.sqrt((np.sum((sadp_data[3151:3501, 5] - sarima_data[3137:3487, 0]) ** 2) / n))
sarima_mape = np.sum(np.abs(sadp_data[3151:3501, 5] - sarima_data[3137:3487, 0])) / n
sarima_picp = 0
sm = 0
for i in range(3137, 3137+m):

    if sarima_data[i, 2] <= sadp_data[i, 5] <= sarima_data[i, 1]:
        sm += 1
sarima_picp = sm/m
pinaw = np.sum(sarima_data[3137:3487, 1] - sarima_data[3137:3487, 2])
sarima_pinaw = pinaw/(R*m)
print('sarima_picp:', sarima_picp, 'sarima_pinaw:', sarima_pinaw)
print('sarima_rmse:', sarima_rmse)
print('sarima_mae:', sarima_mape)
'''
te_data_power = sum([i / 6 for i in sadp_data[:-1, 5]])
sadp_data_power = sum([i / 6 for i in sadp_data[1:, 0]])
sadp_data_power_s = sum([i / 6 for i in sadp_data[1:, 1]])
sadp_data_power_x = sum([i / 6 for i in sadp_data[1:, 2]])

ep_data_power = sum([i / 6 for i in enbpi_data[:-2, 0]])
ep_data_power_s = sum([i / 6 for i in enbpi_data[:-2, 1]])
ep_data_power_x = sum([i / 6 for i in enbpi_data[:-2, 2]])

sari_data_power = sum([(i+200) / 6 for i in sarima_data[:-3, 0]])
sari_data_power_s = sum([(i+200) / 6 for i in sarima_data[:-3, 1]])
sari_data_power_x = sum([(i+200) / 6 for i in sarima_data[:-3, 2]])
'''

te_data_power = sum([i / 6 for i in sadp_data[3151:3501, 5]])
sadp_data_power = sum([i / 6 for i in sadp_data[3152:3502, 0]])
sadp_data_power_s = sum([i / 6 for i in sadp_data[3152:3502, 1]])
sadp_data_power_x = sum([i / 6 for i in sadp_data[3152:3502, 2]])

ep_data_power = sum([i / 6 for i in enbpi_data[3150:3500, 0]])
ep_data_power_s = sum([i / 6 for i in enbpi_data[3150:3500, 1]])
ep_data_power_x = sum([i / 6 for i in enbpi_data[3150:3500, 2]])

sari_data_power = sum([(i+200) / 6 for i in sarima_data[3137:3487, 0]])
sari_data_power_s = sum([(i+200) / 6 for i in sarima_data[3137:3487, 1]])
sari_data_power_x = sum([(i+200) / 6 for i in sarima_data[3137:3487, 2]])

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
# x_normal = np.linspace(int(min(sadp_samples_normal)), 100, int(max(sadp_samples_normal)))  # x轴范围
# y_normal = norm.pdf(x_normal, sadp_mu, sadp_sigma)  # 计算概率密度函数值.
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
plt.show()
