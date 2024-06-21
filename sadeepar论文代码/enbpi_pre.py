import warnings

import numpy as np
import pandas as pd
from matplotlib import pylab as plt
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from mapie.metrics import regression_coverage_score, regression_mean_width_score
from mapie.subsample import BlockBootstrap
from mapie.time_series_regression import MapieTimeSeriesRegressor

warnings.simplefilter("ignore")
# url_file = "https://raw.githubusercontent.com/scikit-learn-contrib/MAPIE/master/examples/data/demand_temperature.csv"
# demand_df = pd.read_csv(
#     url_file, parse_dates=True, index_col=0
# )
demand_df = pd.read_excel(
    "./pre_data/wd-data/T02_2014_filter.xls", usecols=[1, 2, 3, 4])
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


X_train = train_data_all[:-1, :3]
y_train = train_data_all[1:, 3]
X_test = test_data_all[1:, :3]
y_test = test_data_all[1:, 3]
x1 = [i for i in range(len(y_test))]
x2 = [i + len(y_test) for i in range(len(y_train))]
'''
plt.figure(figsize=(16, 5))

plt.plot(x1, y_test)
plt.plot(x2, y_train)
plt.ylabel("power(GW)")
'''
# plt.show()
# Model: Random Forest previously optimized with a cross-validation
model = RandomForestRegressor(max_depth=3, n_estimators=1, random_state=50)


testY_fgy = scaler_y.inverse_transform(y_test.reshape(-1, 1))
# plt.figure(figsize=(16, 5))
# plt.plot(y_data)
# plt.plot(testY_fgy)
# plt.show()

alpha = 0.01
gap = 1
cv_mapiets = BlockBootstrap(
    n_resamplings=10, length=10, overlapping=True, random_state=59
)
mapie_enbpi = MapieTimeSeriesRegressor(
    model, method="enbpi", cv=cv_mapiets, agg_function="mean", n_jobs=-1
)
print("EnbPI with partial_fit, width optimization")
mapie_enbpi.fit(X_train, y_train)

y_pred_npfit, y_pis_npfit = mapie_enbpi.predict(
    X_test, alpha=alpha, ensemble=True, optimize_beta=True
)
y_pred_npfit = scaler_y.inverse_transform(y_pred_npfit.reshape(-1, 1))
y_pis_npfit = scaler_y.inverse_transform(y_pis_npfit.reshape(-1, 2)).reshape(-1, 2, 1)
coverage_npfit = regression_coverage_score(
    y_test, y_pis_npfit[:, 0, 0], y_pis_npfit[:, 1, 0]
)
width_npfit = regression_mean_width_score(
    y_pis_npfit[:, 0, 0], y_pis_npfit[:, 1, 0]
)

y_preds = [y_pred_npfit, y_pred_npfit]
# y_pis = [y_pis_npfit, y_pis_npfit]
coverages = [coverage_npfit, coverage_npfit]
widths = [width_npfit, width_npfit]


def plot_forecast(y_train, y_test, y_preds, y_pis, coverages, widths, plot_coverage=True):
    fig, ax = plt.subplots(
        nrows=1, ncols=1, figsize=(14, 8), sharey="row", sharex="col"
    )
    ax.set_ylabel("power(GW)")
    ax.plot(x1, testY_fgy, lw=2, label="Test data", c="C1")

    ax.plot(
            x1, y_preds[0], lw=2, c="C2", label="Predictions"
        )
    ax.fill_between(
            x1,
            y_pis[:, 0, 0],
            y_pis[:, 1, 0],
            color="C2",
            alpha=0.2,
            label="Prediction intervals",
        )
    title = f"EnbPI, {['with']} update of residuals. "
    if plot_coverage:
        title += f"Coverage:{coverages[0]:.3f} and Width:{widths[0]:.3f}"
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    y_pis = y_pis.reshape(-1, 2)

    dt_arr = np.append(y_pred_npfit, y_pis, axis=1)
    # dt_arr = np.append(dt_arr, y_pis[:, 1], axis=1)
    dt_arr = np.append(dt_arr, testY_fgy, axis=1)
    p_d = pd.DataFrame(dt_arr, columns=['预测值', '一倍下限', '一倍上限', '实际值'])
    p_d.to_csv('11-T02-enbpi-修正风速功率概率预测结果-1步-2.csv', index=False)
    plt.show()


plot_forecast(y_train, testY_fgy, y_preds, y_pis_npfit, coverages, widths)
# model_params_fit_not_done = False
# if model_params_fit_not_done:
#     # CV parameter search
#     n_iter = 100
#     n_splits = 5
#     tscv = TimeSeriesSplit(n_splits=n_splits)
#     random_state = 59
#     rf_model = RandomForestRegressor(random_state=random_state)
#     rf_params = {"max_depth": randint(2, 30), "n_estimators": randint(10, 100)}
#     cv_obj = RandomizedSearchCV(
#         rf_model,
#         param_distributions=rf_params,
#         n_iter=n_iter,
#         cv=tscv,
#         scoring="neg_root_mean_squared_error",
#         random_state=random_state,
#         verbose=0,
#         n_jobs=-1,
#     )
#     cv_obj.fit(X_train, y_train)
#     model = cv_obj.best_estimator_
# else:
#     # Model: Random Forest previously optimized with a cross-validation
#     model = RandomForestRegressor(
#         max_depth=10, n_estimators=50, random_state=59)
# alpha = 0.05
# gap = 1
# cv_mapiets = BlockBootstrap(
#     n_resamplings=100, length=48, overlapping=True, random_state=59
# )
# mapie_enbpi = MapieTimeSeriesRegressor(
#     model, method="enbpi", cv=cv_mapiets, agg_function="mean", n_jobs=-1
# )
# print("EnbPI, with no partial_fit, width optimization")
# mapie_enbpi = mapie_enbpi.fit(X_train, y_train)
# y_pred_npfit, y_pis_npfit = mapie_enbpi.predict(
#     X_test, alpha=alpha, ensemble=True, optimize_beta=True
# )
# coverage_npfit = regression_coverage_score(
#     y_test, y_pis_npfit[:, 0, 0], y_pis_npfit[:, 1, 0]
# )
# width_npfit = regression_mean_width_score(
#     y_pis_npfit[:, 0, 0], y_pis_npfit[:, 1, 0]
# )
# print("EnbPI with partial_fit, width optimization")
# mapie_enbpi = mapie_enbpi.fit(X_train, y_train)
#
# y_pred_pfit = np.zeros(y_pred_npfit.shape)
# y_pis_pfit = np.zeros(y_pis_npfit.shape)
# y_pred_pfit[:gap], y_pis_pfit[:gap, :, :] = mapie_enbpi.predict(
#     X_test.iloc[:gap, :], alpha=alpha, ensemble=True, optimize_beta=True
# )
# for step in range(gap, len(X_test), gap):
#     mapie_enbpi.partial_fit(
#         X_test.iloc[(step - gap):step, :],
#         y_test.iloc[(step - gap):step],
#     )
#     (
#         y_pred_pfit[step:step + gap],
#         y_pis_pfit[step:step + gap, :, :],
#     ) = mapie_enbpi.predict(
#         X_test.iloc[step:(step + gap), :],
#         alpha=alpha,
#         ensemble=True,
#         optimize_beta=True
#     )
# coverage_pfit = regression_coverage_score(
#     y_test, y_pis_pfit[:, 0, 0], y_pis_pfit[:, 1, 0]
# )
# width_pfit = regression_mean_width_score(
#     y_pis_pfit[:, 0, 0], y_pis_pfit[:, 1, 0]
# )
# y_preds = [y_pred_npfit, y_pred_pfit]
# y_pis = [y_pis_npfit, y_pis_pfit]
# coverages = [coverage_npfit, coverage_pfit]
# widths = [width_npfit, width_pfit]