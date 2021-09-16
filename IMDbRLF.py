import keras
from keras.datasets import imdb
import numpy as np
import torch as tc
from collections import  Counter
from collections import Counter
from random import sample


def cul_fre_np(data_list, max_len):
    z = np.zeros((len(data_list), max_len))
    for ik in range(len(data_list)):
        a = np.bincount(data_list[ik], minlength=max_len)
        z[ik, :] = a != 0
    return z


def feature_map(img, data_type, device):
    img0 = tc.tensor(img * np.pi/2, dtype=data_type, device=device)
    img = tc.zeros((img.shape[0], img.shape[1], 2), dtype=data_type, device=device)
    img[:, :, 0] = tc.cos(img0)
    img[:, :, 1] = tc.sin(img0)
    return img


def cul_label(train_data, test_data, batch_num, data_type, device, base):
    total_num = test_data.shape[0]
    circle_num = total_num // batch_num
    yushu = total_num % batch_num
    if yushu != 0:
        inner_product = tc.zeros((len(train_data), total_num - yushu), dtype=data_type, device=device)
        for i in range(circle_num):
            for ii in range(len(train_data)):
                inner_product0 = tc.einsum('nld, mld-> nml', [test_data[(i * batch_num):(i + 1) * batch_num, :, :], train_data[ii]])
                inner_product[ii:, (i * batch_num):(i + 1) * batch_num] = tc.einsum('nm-> n', [base ** tc.einsum('nml-> nm', [tc.log10(inner_product0 + 1e-7)])])/train_data[ii].shape[0]
        yushu_tensor = tc.zeros((len(train_data), yushu), dtype=data_type, device=device)
        for iii in range(len(train_data)):
            inner_product00 = tc.einsum('nld, mld-> nml', [test_data[(total_num - yushu):, :, :], train_data[iii]])
            inner_product_yushu = tc.einsum('nm-> n', [base ** tc.einsum('nml-> nm', [tc.log10(inner_product00 + 1e-7)])])/train_data[iii].shape[0]
            yushu_tensor[iii, :] = inner_product_yushu
        inner_product = tc.cat((inner_product, yushu_tensor), dim=1)
    else:
        inner_product = tc.zeros((len(train_data), total_num), dtype=data_type, device=device)
        for i in range(circle_num):
            for ii in range(len(train_data)):
                inner_product0 = tc.einsum('nld, mld-> nml', [test_data[(i * batch_num):(i + 1) * batch_num, :, :], train_data[ii]])
                inner_product[ii:, (i * batch_num):(i + 1) * batch_num] = tc.einsum('nm-> n', [base ** tc.einsum('nml-> nm', [tc.log10(inner_product0 + 1e-7)])])/train_data[ii].shape[0]
    label_predict = tc.argmax(inner_product, dim=0)
    return label_predict


data_type = tc.float32
device = 'cuda:0'
base = 1.07
batch_num = 50
num_words = 700
epochs = 30
pic_num = 'all'

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=700, skip_top=60, maxlen=100)
data_train_list = []
test_data_list = []
for x in range(train_data.shape[0]):
    data_train_list.append(np.array(train_data[x]))
for xx in range(test_data.shape[0]):
    test_data_list.append(np.array(test_data[xx]))
train_labels = tc.tensor(train_labels, dtype=data_type, device=device)
test_labels = tc.tensor(test_labels, dtype=data_type, device=device)

b = tc.tensor(cul_fre_np(data_train_list, num_words), dtype=data_type, device=device)
c = tc.tensor(cul_fre_np(test_data_list, num_words), dtype=data_type, device=device)
print(b.shape, c.shape)
test_data = feature_map(c, data_type, device)
class0_train_data = (b[train_labels == 0])
class1_train_data = (b[train_labels == 1])

if pic_num == 'all':
    list_train_data = [feature_map(class0_train_data, data_type, device), feature_map(class1_train_data, data_type, device)]
    label_predict = cul_label(list_train_data, test_data, batch_num, data_type, device, base)
    accuracy = tc.sum(test_labels == label_predict, dim=0).item() / label_predict.shape[0]
    print('Acc:', accuracy)
else:
    num0 = list(sample(range(class0_train_data.shape[0]), pic_num))
    num1 = list(sample(range(class1_train_data.shape[0]), pic_num))
    data0 = class0_train_data[num0]
    data1 = class1_train_data[num1]
    data0 = feature_map(data0, data_type, device)
    data1 = feature_map(data1, data_type, device)
    label_predict = cul_label([data0, data1], test_data, batch_num, data_type, device, base)
    accuracy = tc.sum(test_labels == label_predict, dim=0).item()/label_predict.shape[0]
    print('Acc:', accuracy)