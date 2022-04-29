# PD-SECR
Ethereum  Ponzi scheme Detection based on PD-SECR


#Refer to Github for some code

https://github.com/xuyl0104/blockchain_ponzi_detection

#The download address of the dataset

https://drive.google.com/open?id=1izaOs4Mlp6dxdRMtRYQeUfkDhlqLf4Z6
             
Code execution steps：
（1）run prepare_dataset.py
（2）run train-cnn.py
（3）run train-RF.py:
Before running train-RF.py, modify the loading model path first, and load the CNN model with the best training to RF.
model = torch.load('./runs/2022-04-29-15-14-50/best.pt'), cnn each training is different. best.pt files generated are in the runs folder, and best.pt paths need to be transferred.
 for example :model = torch.load('./runs/2022-04-29-15-14-50/best.pt')
