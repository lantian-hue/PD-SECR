import os
import pickle
import pandas as pd
import numpy as np
from numpy import save
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN,SMOTETomek
from imblearn.over_sampling import SVMSMOTE,SMOTEN,BorderlineSMOTE,SMOTENC

test_size = 0.2
is_balance = False
# is_balance = True
# 数据处理
data = pd.read_csv('./dataset/features2.csv')  #读取csv文件
labels = data['ponzi']
features = data.drop(['address', 'ponzi'], 1)

if is_balance:
    sme = SMOTEENN(random_state=42)
    features,labels=sme.fit_resample(features,labels)
    save = pd.DataFrame(features, columns=['return_rio', 'A_bal', 'investments_num','payments_num','Pr','maxpay','D_ind','GASLIMIT',
                                       'EXP','CALLDATALOAD','SLOAD','CALLER','LT','GAS','MOD','MSTORE'])
   # save1 = pd.DataFrame(labels, columns=['ponzi'])
   # save.to_csv("D:\\Thesis experiment\\dataset\\features1")
   # save1.to_csv("D:\\Thesis experiment\\dataset\\labels1")


x_train, x_test, y_train, y_test = train_test_split(np.array(features), np.array(labels), test_size=test_size, shuffle=True, random_state=123)  #划分训练集和测试集
#if is_balance:
#    sme = SMOTEENN(random_state=42)
#   x_train, y_train = sme.fit_resample(x_train, y_train)

x_train = np.expand_dims(x_train, axis=1)
x_test = np.expand_dims(x_test, axis=1)#扩展维度



if is_balance:
    with open('./dataset/cnn_balance_data_'+str(test_size)+'.pkl', 'wb') as f:
        pickle.dump({'x_train':x_train, 'y_train':y_train, 'x_test':x_test, 'y_test':y_test}, f)
        #pickle模块实现了基本的数据序列和反序列化
else:
    with open('./dataset/cnn_data_'+str(test_size)+'.pkl', 'wb') as f:
        pickle.dump({'x_train':x_train, 'y_train':y_train, 'x_test':x_test, 'y_test':y_test}, f)
