# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Flatten
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import re
import cv2
import os

from build_label import data_label
from VGG_16 import VGG16_model
from ZFNet import ZFNet_model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

magnification = '40X'
total_acc = 0
for i in range(1,6):   #cross_validation
    current_path = os.path.join('../../BC_Fold_Data/fold%d/' %i)
    train_path = os.path.join('%strain/' % current_path)
    test_path = os.path.join('%stest/' %current_path)
    train_path_B = os.path.join('%sB/%s/' %(train_path, magnification))
    train_path_M = os.path.join('%sM/%s/' % (train_path, magnification))

   #train
    train_data, train_label = data_label(train_path_B, train_path_M)
    cnn_model = ZFNet_model()
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    cnn_model.fit(train_data, train_label, batch_size=32, epochs=100, shuffle=True,
              validation_split=0.1, callbacks=[early_stopping])

    #test
    pathDir = os.listdir(test_path)
    acc = 0
    cnt = 0
    for patient in pathDir:
        if patient[0]!='S':     #not patient folders
            continue
        patient_path = os.path.join('%s%s/%s/'%(test_path, patient, magnification))
        if patient[4]=='B':
            test_data, test_label = data_label(patient_path, None)
        else:
            test_data, test_label = data_label(None, patient_path)
        loss, accuracy = cnn_model.evaluate(test_data, test_label)
        print patient,accuracy
        acc = acc + accuracy
        cnt = cnt + 1

    acc=acc/cnt
    print current_path,acc
    total_acc = total_acc + acc

total_acc = total_acc/5
print 'final patient score: ', total_acc





