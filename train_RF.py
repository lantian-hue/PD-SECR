# import os
# import numpy as np
import pickle

import matplotlib
import sklearn
import torch
# from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import seaborn as sns
# import lightgbm as lgt
from sklearn.model_selection import KFold, cross_validate





def plot_confusion_matrix(y_true, y_pred):
    # x=sklearn.metrics.confusion_matrix(y_true, y_pred, labels=None, sample_weight=None)
    # print(x)
    sns.set()
    f,ax=plt.subplots()
    C2= confusion_matrix(y_true, y_pred, labels=range(2))
    #
    matplotlib.rc('figure', figsize=(14, 7))
    matplotlib.rc('font', size=14)
    matplotlib.rc('axes', grid=False)
    matplotlib.rc('axes', facecolor='white')

    # 设置字体为楷体
    matplotlib.rcParams['font.sans-serif'] = ['KaiTi']

    sns.heatmap(C2,annot=True,fmt='.20g',ax=ax) #画热力图
    ax.set_title('混淆矩阵') #标题
    ax.set_xlabel('预测') #x轴
    ax.set_ylabel('真实') #y轴
    plt.show()

#

# 加载模型
model = torch.load('./runs/2022-04-08-09-25-53/best.pt')
model.eval()
if torch.cuda.is_available():
    model.cuda()

test_size = 0.2
#is_balance = False
is_balance = True
# 加载数据
if is_balance:
    with open('./dataset/cnn_balance_data_'+str(test_size)+'.pkl', 'rb') as f:
        data = pickle.load(f)
else:
    with open('./dataset/cnn_data_'+str(test_size)+'.pkl', 'rb') as f:
        data = pickle.load(f)
x_train = data['x_train']
x_test = data['x_test']
y_train = data['y_train']
y_test = data['y_test']

x_train = torch.from_numpy(x_train).float()
x_test = torch.from_numpy(x_test).float()
if torch.cuda.is_available():
    x_train = x_train.cuda()
    x_test = x_test.cuda()

# 获取特征
train_feature = model.get_feature(x_train)
test_feature = model.get_feature(x_test)

train_feature = train_feature.cpu().detach().numpy()
test_feature = test_feature.cpu().detach().numpy()

precision1=0
recall1 = 0
accuracy1 = 0
f11 = 0
number = 0

# 训练svm
# clf = SVC()
# clf.fit(train_feature, y_train)
clf = RandomForestClassifier()
clf.fit(train_feature,y_train)
# clf = lgt.LGBMClassifier()
# clf.fit(train_feature,y_train)

# 保存svm模型
with open('RF.pkl', 'wb') as f:
    pickle.dump(clf, f)
# with open('svm.pkl', 'wb') as f:
#     pickle.dump(clf, f)
# with open('lgt.pkl', 'wb') as f:
#     pickle.dump(clf, f)

pred = clf.predict(test_feature)
recall = recall_score(y_test, pred)
precision = precision_score(y_test,pred)
accuracy = accuracy_score(y_test, pred)
f1 = f1_score(y_test, pred)

print('precision:{:.2f},recall:{:.2f},f1:{:.2f},accuracy:{:.2f}'.format(precision,recall,f1,accuracy))

plot_confusion_matrix(y_test, pred) #
