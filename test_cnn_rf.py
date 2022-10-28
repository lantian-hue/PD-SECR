import os
import numpy as np
import pickle
import torch


# 加载cnn模型
if torch.cuda.is_available():
    cnn = torch.load('./runs/2022-04-02-20-37-41/best.pt')
    cnn.cuda()
else:
    cnn = torch.load('./runs/2022-04-02-20-37-41/best.pt', map_location='cpu')
cnn.eval()

# 加载svm模型
with open('RF.pkl', 'rb') as f:
    RF = pickle.load(f)

# 加载数据
with open('./dataset/cnn_balance_data_0.2.pkl', 'rb') as f:
    cnn_data = pickle.load(f)

x_test = cnn_data['x_test']
y_test = cnn_data['y_test']
print(x_test,y_test)


for i in range(x_test.shape[0]):
    input = np.expand_dims(x_test[i], axis=0)
    input = torch.from_numpy(input).float()
    if torch.cuda.is_available():
        input = input.cuda()

    feature = cnn.get_feature(input)      # cnn提取特征
    feature = feature.cpu().detach().numpy()
    pred = RF.predict(feature)
    print(pred)
