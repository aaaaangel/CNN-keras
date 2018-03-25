# -*- coding: utf-8 -*-
from keras.utils import np_utils
import numpy as np
import re
import cv2
import os

def getnum(file_path):
    if file_path==None:
        return 0
    pathDir = os.listdir(file_path)
    i = 0
    for allDir in pathDir:
        i +=1
    return i

def data_label(path_b, path_m):
    count = getnum(path_b) + getnum(path_m)
    data_list = np.empty((count, 460, 700, 3), dtype='float32')
    label_list = np.empty((count,),dtype = 'uint8')

    index = 0
    if path_b!=None:
        pathDir = os.listdir(path_b)
        for each_image in pathDir:
            img_path = os.path.join('%s%s' % (path_b, each_image))  # 路径进行连接
            image = cv2.imread(img_path, -1)
            rows, cols, channels = image.shape
            if cols!=460:
                image = cv2.resize(image, (700,460))

            array = np.asarray(image, dtype='float32')
            array -= np.min(array)
            array /= np.max(array)
            data_list[index, :, :, :] = array
            label_list[index] = 0
            index += 1

    if path_m!=None:
        pathDir = os.listdir(path_m)
        for each_image in pathDir:
            img_path = os.path.join('%s%s' % (path_m, each_image))  # 路径进行连接
            image = cv2.imread(img_path, -1)
            rows, cols, channels = image.shape
            if cols!=460:
                image = cv2.resize(image, (700,460))

            array = np.asarray(image, dtype='float32')
            array -= np.min(array)
            array /= np.max(array)
            data_list[index, :, :, :] = array
            label_list[index] = 1
            index += 1

    permutation = np.random.permutation(data_list.shape[0])
    shuffled_data_list = data_list[permutation, :, :, :]
    shuffled_label_list = label_list[permutation,]

    shuffled_label_list = np_utils.to_categorical(shuffled_label_list, 2)

    print len(data_list), len(label_list)
    return shuffled_data_list, shuffled_label_list