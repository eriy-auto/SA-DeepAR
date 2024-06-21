import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import normal_ad
from scipy import stats
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

import pmdarima as pm
pd.plotting.register_matplotlib_converters()

# ---- Data Transformations ----

'''
months_dict = {'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,
               'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12}

df = pd.read_csv('D:/PYTHON_summary/attention_deepar/pre_data/station_sao_paulo.csv')
df = df[['YEAR'] + list(months_dict.keys())]

df_sp = pd.melt(df,
        id_vars='YEAR',
        value_vars=months_dict.keys(),
        var_name='MONTH',
        value_name='Sum of Value').replace({"MONTH": months_dict})

df_sp['DAY'] = 1 #need a day for the pd.to_datetime() function
df_sp['DATE'] = pd.to_datetime(df_sp[['YEAR','MONTH','DAY']])
df_sp = df_sp[['DATE','Sum of Value']].rename(columns={'DATE':'date','Sum of Value':'temp'})
df_sp = df_sp.sort_values('date').reset_index(drop=True)

'''
# ---- Visualize Data ----

'''
plt.figure(figsize=(15,7))
plt.title("Sao Paulo AVG Monthly Temperature - w/ Err")
plt.plot(df_sp["date"], df_sp['temp'], color='#379BDB', label='Original')
plt.show(block=False)


df_sp = df_sp[(df_sp['date'] > '1982-03-01') & (df_sp['date'] < '2013-04-01')]
df_sp = df_sp.set_index(df_sp['date'], drop=True).drop(columns=["date"])

train, valid = df_sp[:int(len(df_sp)*0.8)], df_sp[int(len(df_sp)*0.8):]

plt.figure(figsize=(15,7))
plt.title("Sao Paulo AVG Monthly Temperature - 1982-03-01 to 2013-04-01")
plt.plot(train.index, train['temp'], color='#379BDB', label='Train')
plt.plot(valid.index, valid['temp'], color='#fc7d0b', label='Valid')
plt.xlabel('Date')
plt.ylabel('Temp - Celsius')
plt.legend()
plt.show(block=False)
'''

demand_df = pd.read_excel(
    "./pre_data/wd-data/T01_2014_filter.xls", usecols=[1, 2, 3, 4])
use_data = demand_df.to_numpy()
index_f = np.where(use_data[:, 3] < 0)
m_data = np.delete(use_data, index_f, axis=0)
print(m_data.shape)
x_data = m_data[20000:40000, :3].reshape(-1, 3)
y_data = m_data[20000:40000, 3].reshape(-1, 1)
scaler_x = MinMaxScaler(feature_range=(0, 1))
x_data_scaler = scaler_x.fit_transform(x_data)
scaler_y = MinMaxScaler(feature_range=(0, 1))
y_data_scaler = scaler_y.fit_transform(y_data)
all_data_scaler = np.append(x_data_scaler, y_data_scaler, axis=1)

print('all_data_scaler', all_data_scaler.shape)
# print('all_data_scaler_2', all_data_scaler_2.shape)
train_data_all = all_data_scaler[5000:, :]
test_data_all = all_data_scaler[:5000, :]

# '轮毂转速', '叶片角度', '风速', '功率'


X_train = train_data_all[:, :3]
y_train = train_data_all[:, 3]
X_test = test_data_all[:, :3]
y_test = test_data_all[:, 3]


'''
SARIMA_model = pm.auto_arima(train['temp'], start_p=1, start_q=1,
                         test='adf',
                         max_p=3, max_q=3,
                         m=12, #annual frequency(12 for month, 7 for week etc)
                         start_P=0,
                         seasonal=True, #set to seasonal
                         d=None,
                         D=1, #order of the seasonal differencing
                         trace=False,
                         error_action='ignore',
                         suppress_warnings=True,
                         stepwise=True)
'''
testY_fgy = scaler_y.inverse_transform(y_test.reshape(-1, 1))

arima_preds = []

m_sarima = ARIMA(y_train, exog=X_train, order=(1, 1, 1), seasonal_order=(0, 1, 2, 12)).fit()
arima_preds.append(m_sarima.forecast(X_test.shape[0], exog=X_test))

residuals = sorted([x - y for x, y in zip(arima_preds, y_test[:10])])
idx = [i for i in range(y_test.shape[0])]

sw_result = stats.shapiro(residuals)
ad_result = normal_ad(np.array(residuals), axis=0)
# dag_result = stats.normaltest(residuals, axis=0, nan_policy='propagate')

'''
plt.figure(figsize=(15,7))
res = stats.probplot(residuals, plot=plt)
ax = plt.gca()
ax.annotate("SW p-val: {:.4f}".format(sw_result[1]), xy=(0.05,0.9), xycoords='axes fraction', fontsize=15,
            bbox=dict(boxstyle="round", fc="none", ec="gray", pad=0.6))
ax.annotate("AD p-val: {:.4f}".format(ad_result[1]), xy=(0.05,0.8), xycoords='axes fraction', fontsize=15,
            bbox=dict(boxstyle="round", fc="none", ec="gray", pad=0.6))
ax.annotate("DAG p-val: {:.4f}".format(dag_result[1]), xy=(0.05,0.7), xycoords='axes fraction', fontsize=15,
             bbox=dict(boxstyle="round", fc="none", ec="gray", pad=0.6))

plt.show()
'''

RMSFE = np.sqrt(sum([x**2 for x in residuals]) / len(residuals))
band_size = 0.8*RMSFE # 1.96;0.8

s_b = arima_preds-band_size
x_b = arima_preds+band_size
pre_d = np.array(arima_preds[0])
y_pred_s = scaler_y.inverse_transform(s_b.reshape(-1, 1)).ravel()
y_pred_x = scaler_y.inverse_transform(x_b.reshape(-1, 1)).ravel()
y_pred = scaler_y.inverse_transform(pre_d.reshape(-1, 1))

fig, ax = plt.subplots(figsize=(15,7))
ax.plot(idx, testY_fgy, color='#fc7d0b', label='Valid')
ax.plot(idx, y_pred, color='b', label='Predict')
ax.fill_between(idx, y_pred_x, y_pred_s, color='b', alpha=.1)
ax.set_title("Predictions w/ 95% Confidence")
ax.set_xlabel('Date')
ax.set_ylabel('Temp - Celsius')
plt.legend()

y_pred_x = y_pred_x.reshape(-1, 1)
y_pred_s = y_pred_s.reshape(-1, 1)
dt_arr = np.append(y_pred, y_pred_x, axis=1)
dt_arr = np.append(dt_arr, y_pred_s, axis=1)
dt_arr = np.append(dt_arr, testY_fgy, axis=1)
p_d = pd.DataFrame(dt_arr, columns=['预测值', '一倍下限', '一倍上限', '实际值'])
p_d.to_csv('11-T02-sarima-修正风速功率概率预测结果-1步-2.csv', index=False)
plt.show()