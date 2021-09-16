import keras
from keras.datasets import imdb
import numpy as np
import torch as tc
from collections import  Counter
from random import sample


def cul_fre_np(data_list, max_len):
    z = np.zeros((len(data_list), max_len))
    for ik in range(len(data_list)):
        a = np.bincount(data_list[ik], minlength=max_len)
        z[ik, :] = a != 0
    return z


def cul_distance_with_cluster_and_picture_set(centroid_tensor, picture_set, sigma):
    b = tc.zeros((centroid_tensor.shape[0], picture_set.shape[0]), dtype=tc.float32, device='cuda:0')
    for k in range(centroid_tensor.shape[0]):
        b[k, :] = tc.exp((-1) * tc.norm(picture_set - centroid_tensor[k], 2, dim=1)/(2 * sigma))
    return b


def cul_acu(label_predict, label_test):
    accuracy = ((label_predict == label_test).nonzero().squeeze().shape[0] / label_predict.shape[0])
    return accuracy


def cul_distance(centroid_tensor, centroid_tensor_lable, train_set, sigma, batch_num=4000):
    total_num = train_set.shape[0]
    circle_num = total_num // batch_num
    yushu = total_num % batch_num
    label_predict = tc.zeros((centroid_tensor.shape[0], train_set.shape[0]), dtype=tc.float32, device='cuda:0')
    if total_num % batch_num != 0:
        for i in range(circle_num):
            label_predict[:, (i * batch_num):(i + 1) * batch_num] = \
                cul_distance_with_cluster_and_picture_set(centroid_tensor, train_set[(i * batch_num):(i + 1) * batch_num, :], sigma)
        label_predict[:, total_num - yushu:] = \
            cul_distance_with_cluster_and_picture_set(centroid_tensor, train_set[(total_num - yushu):, :], sigma)
    else:
        for i in range(circle_num):
            label_predict[:, (i * batch_num):(i + 1) * batch_num] = \
                cul_distance_with_cluster_and_picture_set(centroid_tensor, train_set[(i * batch_num):(i + 1) * batch_num, :], sigma)
    a0 = label_predict[centroid_tensor_lable == 0, :]
    b0 = label_predict[centroid_tensor_lable == 1, :]
    a = tc.sum(a0, dim=0, keepdim=True)
    b = tc.sum(b0, dim=0, keepdim=True)
    return tc.topk(tc.cat((a/a0.shape[0], b/b0.shape[0]), dim=0), dim=0, k=1, largest=True)[1]


data_type = tc.float32
device = 'cuda:0'
batch_num = 50
num_words = 700
epochs = 30
sigma = 0.1
pic_num = 'all'

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=num_words, skip_top=60, maxlen=100)

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

if pic_num == 'all':
    lable_predict = tc.squeeze(cul_distance(b, train_labels, c, sigma))
    accuracy = cul_acu(lable_predict, test_labels)
    print('Acc:', accuracy)
else:
    class0_train_data = (b[train_labels == 0])
    class1_train_data = (b[train_labels == 1])

    num0 = list(sample(range(class0_train_data.shape[0]), pic_num))
    num1 = list(sample(range(class1_train_data.shape[0]), pic_num))
    centroid_tensor = tc.cat((class0_train_data[num0], class1_train_data[num1]), dim=0)
    centroid_tensor_label = tc.cat((tc.zeros(len(num0), dtype=data_type, device=device), tc.ones(len(num1), dtype=data_type, device=device)), dim=0)
    lable_predict = tc.squeeze(cul_distance(centroid_tensor, centroid_tensor_label, c, sigma))
    accuracy = cul_acu(lable_predict, test_labels)
    print('Acc:', accuracy)

