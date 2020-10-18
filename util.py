import xlrd
import numpy as np
from sklearn import preprocessing
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
import os
import pandas as pd

root_directory = os.path.dirname(__file__)
all_energy_data_path = root_directory + '/data/Energy Final Data.csv'
energy_return_data_path = root_directory + '/data/Energy Final Data -Return series.xlsx'
energy_data_after_corona_path = root_directory + '/data/Energy Data-After Corona.xlsx'


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


def getdata_energy_after_Corona(window, time_difference=True):
    wb = xlrd.open_workbook(energy_data_after_corona_path)
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

    np_data = np_data[:, [1, 7, 8, 9, 10]]
    col_mean = np.nanmean(np_data, axis=0)
    inds = np.where(np.isnan(np_data))
    np_data[inds] = np.take(col_mean, inds[1])
    # print(np.argwhere(np.isnan(np_data)))
    print(np.isnan(np.sum(np_data)))
    if time_difference:
        orig = np_data[1:, -1]
        lagged = np_data[:-1, -1]
        time_differenced = orig - lagged
        np_data = np_data[1:, :]
        np_data[:, -1] = time_differenced

    print(np_data[0])

    scaler = preprocessing.StandardScaler(with_mean=True, with_std=True)
    # scaler = preprocessing.Normalizer()
    # scaler = preprocessing.MinMaxScaler()
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
    return windowed, np_data


def plot_result(y_true,y_pred ):
    xx = np.array(range(len(y_true)))
    # print(len(xx))

    plt.plot(xx, y_true, label="Truth")
    plt.plot(xx, y_pred, label="Prediction")
    plt.title('output')
    # plt.legend()
    plt.legend(loc="upper left")
    # plt.savefig("fig.jpg")
    plt.show()


def plot_loss(losses, losses2):
    xx = np.array(range(len(losses)))
    plt.plot(xx, losses, label="Val losses")
    plt.plot(xx, losses2, label="Train losses")
    plt.title('losses')
    plt.legend(loc="upper left")
    plt.savefig("losses.jpg")
    plt.show()


def get_dataloaders(WINDOW, data_funtion, BATCH_SIZE, device, shuffle=False):
    data, _, scaler = data_funtion(window=WINDOW)
    d_input = np.shape(data)[-1] - 1  # From dataset
    d_output = 1  # From dataset

    # data = data[:300]
    data = data.astype(np.float32)
    n_data = len(data)
    train_data = data[:int(0.8 * n_data)]
    val_data = data[int(0.8 * n_data):int(0.9 * n_data)]
    test_data = data[int(0.9 * n_data):]
    # data = torch.from_numpy(data,device=device)
    all_data = torch.tensor(data, device=device)
    train_data = torch.tensor(train_data, device=device)
    val_data = torch.tensor(val_data, device=device)
    test_data = torch.tensor(test_data, device=device)

    train_dataloader = DataLoader(train_data,
                                  batch_size=BATCH_SIZE,
                                  shuffle=shuffle
                                  )

    val_data = DataLoader(val_data,
                          batch_size=BATCH_SIZE,
                          shuffle=False
                          )
    test_data = DataLoader(test_data,
                           batch_size=BATCH_SIZE,
                           shuffle=False
                           )

    all_data = DataLoader(all_data,
                          batch_size=BATCH_SIZE,
                          shuffle=False
                          )
    return train_dataloader, val_data, test_data, all_data, d_input, d_output, scaler


def all_energy_data(window, time_difference=False):
    df = pd.read_csv(all_energy_data_path)
    # wb = xlrd.open_workbook(all_energy_data_path)
    # sheet = wb.sheet_by_index(0)
    # n = 188
    n, _ = df.shape
    # n = sheet.nrows
    data = []
    for i in range(2, n):
        row = []
        for j in range(1, 19):
            row.append(df.iloc[i, j])
        # print(row)
        data.append(row)
    np_data = np.array(data, dtype=np.float64)
    # print(np.sum(np_data))
    # print(np_data[:10,:])
    np_data = np_data[:, [0, 1, 12, 13, 14, 15, 16]]
    col_mean = np.nanmean(np_data, axis=0)
    inds = np.where(np.isnan(np_data))
    np_data[inds] = np.take(col_mean, inds[1])
    # print(np.argwhere(np.isnan(np_data)))
    print(np.isnan(np.sum(np_data)))
    if time_difference:
        orig = np_data[1:, -1]
        lagged = np_data[:-1, -1]
        time_differenced = orig - lagged
        np_data = np_data[1:, :]
        np_data[:, -1] = time_differenced

    print(np_data[0])
    scaler1 = preprocessing.MinMaxScaler()
    # scaler1 = preprocessing.StandardScaler(with_mean=True, with_std=True)
    scaler2 = preprocessing.StandardScaler(with_mean=True, with_std=True)

    # scaler = preprocessing.Normalizer()
    # scaler = preprocessing.MinMaxScaler()
    np_data = scaler1.fit_transform(np_data)
    # np_data[:, :-1] = scaler2.fit_transform(np_data[:, :-1])
    # np_data[:, -1:] = scaler1.fit_transform(np_data[:, -1:])
    # np_data[:, -1] = np_data[:, -1] / 5
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
    return windowed, np_data, scaler1


def energy_return_data(window):
    wb = xlrd.open_workbook(energy_return_data_path)
    sheet = wb.sheet_by_index(0)
    n = sheet.nrows
    data = []
    for i in range(2, n):
        row = []
        for j in range(18):
            row.append(sheet.cell_value(i, j))
        data.append(row)
    np_data = np.array(data, dtype=np.float64)
    np_data = np_data[:, [2, 13, 14, 15, 16, 17]]
    col_mean = np.nanmean(np_data, axis=0)
    inds = np.where(np.isnan(np_data))
    np_data[inds] = np.take(col_mean, inds[1])
    # print(np.argwhere(np.isnan(np_data)))
    print(np.isnan(np.sum(np_data)))
    # if time_difference:
    #     orig = np_data[1:, -1]
    #     lagged = np_data[:-1, -1]
    #     time_differenced = orig - lagged
    #     np_data = np_data[1:, :]
    #     np_data[:, -1] = time_differenced

    print(np_data[0])
    scaler1 = preprocessing.MinMaxScaler()
    # scaler1 = preprocessing.StandardScaler(with_mean=True, with_std=True)
    scaler2 = preprocessing.StandardScaler(with_mean=True, with_std=True)

    # scaler = preprocessing.Normalizer()
    # scaler = preprocessing.MinMaxScaler()
    np_data = scaler1.fit_transform(np_data)
    # np_data[:, :-1] = scaler2.fit_transform(np_data[:, :-1])
    # np_data[:, -1:] = scaler1.fit_transform(np_data[:, -1:])
    # np_data[:, -1] = np_data[:, -1] / 5
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
    return windowed, np_data, scaler1


from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

eps = 0.001


def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true)))


def MFE(y_true, y_pred):
    return np.mean(y_true - y_pred)


def full_report(y_true, y_pred):
    print('MSE', mean_squared_error(y_true, y_pred))
    print('MAE', mean_absolute_error(y_true, y_pred))
    # print('MAPE', MAPE(y_true, y_pred))
    print('MFE', MFE(y_true, y_pred))
    # print('r2_score', r2_score(y_true, y_pred))
