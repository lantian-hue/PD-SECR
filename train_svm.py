# import os
# import numpy as np
import pickle
import torch
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, recall_score,precision_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
# import lightgbm as lgt
from sklearn.model_selection import KFold, cross_validate


def plot_confusion_matrix(y_true, y_pred):
    sns.set()
    f,ax=plt.subplots()
    C2= confusion_matrix(y_true, y_pred, labels=range(2))
    sns.heatmap(C2,annot=True,fmt='.20g',ax=ax) #画热力图

    ax.set_title('confusion matrix') #标题
    ax.set_xlabel('pred') #x轴
    ax.set_ylabel('true') #y轴
    plt.show()

#

# 加载模型
model = torch.load('./runs/2022-03-04-13-16-37/SMOTEENN-2/best.pt')
model.eval()
if torch.cuda.is_available():
    model.cuda()

test_size = 0.2
is_balance = True
# 加载数据
if is_balance:
    with open('./data/cnn_balance_data_'+str(test_size)+'.pkl', 'rb') as f:
        data = pickle.load(f)
else:
    with open('./data/cnn_data_'+str(test_size)+'.pkl', 'rb') as f:
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


for i in range(21):
    # 训练svm
    clf = SVC()
    clf.fit(train_feature, y_train)
    # clf = RandomForestClassifier()
    # clf.fit(train_feature, y_train)
    # clf = lgt.LGBMClassifier()
    # clf.fit(train_feature,y_train)

    # 保存svm模型
    # with open('RF.pkl', 'wb') as f:
    #     pickle.dump(clf, f)
    with open('svm.pkl', 'wb') as f:
        pickle.dump(clf, f)
    # with open('lgt.pkl', 'wb') as f:
    #     pickle.dump(clf, f)

    pred = clf.predict(test_feature)
    recall = recall_score(y_test, pred)
    precision = precision_score(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)



    print('precision:{:.2f},recall:{:.2f},f1:{:.2f},accuracy:{:.2f}'.format(precision, recall, f1, accuracy))

    # plot_confusion_matrix(y_test, pred)



