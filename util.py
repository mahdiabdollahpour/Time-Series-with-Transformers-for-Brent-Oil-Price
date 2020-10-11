import xlrd
import numpy as np
from sklearn import preprocessing
from matplotlib import pyplot as plt

def save_data():
    wb = xlrd.open_workbook('data-final.xlsx')
    sheet = wb.sheet_by_index(0)
    n = sheet.nrows
    data = {}
    data['Z'] = []
    data['X'] = []
    data['R'] = []
    for i in range(2, n):
        row = []
        for j in range(11):
            row.append(sheet.cell_value(i, j))
        data['Z'].append(row)
        data['X'].append([sheet.cell_value(i, 11)])
    np_data = np.array(data)
    # print(np_data[:10,:])
    np.savez('dataset', np_data)


def getdata(window):
    wb = xlrd.open_workbook('data/data-final.xlsx')
    sheet = wb.sheet_by_index(0)
    n = sheet.nrows
    data = []
    for i in range(2, n):
        row = []
        for j in range(12):
            row.append(sheet.cell_value(i, j))
        data.append(row)
    np_data = np.array(data, dtype=np.float64)
    # print(np.sum(np_data))
    # print(np_data[:10,:])

    col_mean = np.nanmean(np_data, axis=0)
    inds = np.where(np.isnan(np_data))
    np_data[inds] = np.take(col_mean, inds[1])
    # print(np.argwhere(np.isnan(np_data)))
    print(np.isnan(np.sum(np_data)))
    print(np_data[0])
    scaler = preprocessing.StandardScaler()
    np_data[:, :-1] = scaler.fit_transform(np_data[:, :-1])
    np_data[:, -1] = np_data[:, -1] - 1
    print(np_data[0])
    n_data = len(np_data)
    windowed = []
    for i in range(n_data - window + 1):
        seq = []
        for j in range(window):
            # print(i+j,len(data),n_data,window,n_data-window)
            seq.append(np_data[i + j])
        windowed.append(seq)
    windowed = np.array(windowed)
    return windowed


def getdata_energy_after_Corona(window):
    wb = xlrd.open_workbook('data/Energy Data-After Corona.xlsx')
    sheet = wb.sheet_by_index(0)
    n = 188
    # n = sheet.nrows
    data = []
    for i in range(2, n):
        row = []
        for j in range(12):
            row.append(sheet.cell_value(i, j))
        # print(row)
        data.append(row)
    np_data = np.array(data, dtype=np.float64)
    # print(np.sum(np_data))
    # print(np_data[:10,:])
    np_data = np_data[:, [1, 0, 7, 8, 9, 10]]
    col_mean = np.nanmean(np_data, axis=0)
    inds = np.where(np.isnan(np_data))
    np_data[inds] = np.take(col_mean, inds[1])
    # print(np.argwhere(np.isnan(np_data)))
    print(np.isnan(np.sum(np_data)))
    print(np_data[0])
    # scaler = preprocessing.StandardScaler(with_mean=True, with_std=True)
    # scaler = preprocessing.Normalizer()
    scaler = preprocessing.MinMaxScaler()
    np_data = scaler.fit_transform(np_data)
    # np_data = np_data
    print(np_data[0])
    n_data = len(np_data)
    windowed = []
    for i in range(n_data - window + 1):
        seq = []
        for j in range(window):
            # print(i+j,len(data),n_data,window,n_data-window)
            seq.append(np_data[i + j])
        windowed.append(seq)
    windowed = np.array(windowed)
    return windowed

def plot_result(y_pred,y_true):
    # plt.figure(figsize=(30, 30))
    print(np.shape(y_pred))
    print(np.shape(y_true))
    # plt.subplot(10, len(dataloader))
    xx = np.array(range(len(y_true)))
    # print(xx)
    plt.plot(xx, y_true, label="Truth")
    plt.plot(xx, y_pred, label="Prediction")
    plt.title('output')
    # plt.legend()
    plt.savefig("fig.jpg")
    plt.show()
