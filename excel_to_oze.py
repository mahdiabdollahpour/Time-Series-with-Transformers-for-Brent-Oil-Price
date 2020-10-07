import xlrd
import numpy as np


def save_data():
    wb = xlrd.open_workbook('data-final.xlsx')
    sheet = wb.sheet_by_index(0)
    n = sheet.nrows
    data = {}
    data['Z'] = []
    data['X'] = []
    data['R'] = []
    for i in range(2,n):
        row = []
        for j in range(11):
            row.append(sheet.cell_value(i,j))
        data['Z'].append(row)
        data['X'].append([sheet.cell_value(i,11)])
    np_data = np.array(data)
    # print(np_data[:10,:])
    np.savez('dataset',np_data)
def getdata():
    wb = xlrd.open_workbook('data-final.xlsx')
    sheet = wb.sheet_by_index(0)
    n = sheet.nrows
    data = []
    for i in range(2,n):
        row = []
        for j in range(12):
            row.append(sheet.cell_value(i,j))
        data.append(row)

    np_data = np.array(data)
    return np_data

