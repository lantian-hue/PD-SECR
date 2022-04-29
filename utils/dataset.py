import os
import cv2
from tqdm import tqdm
from PIL import Image
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset


# class vggDataset(Dataset):
#     def __init__(self, dir, transform=None):
#         self.label_map = {'DN':0, 'SC':1, 'TBF':2}
#         self.img_files = glob(os.path.join(dir, '*', '*.jpg'))
#         self.transform = transform
#
#     def __getitem__(self, index):
#         img_file = self.img_files[index]
#         img_path, img_name = os.path.split(img_file)
#         _, img_dir = os.path.split(img_path)
#         label = self.label_map[img_dir]
#
#         img = cv2.imread(img_file)
#         h, w = img.shape[:2]
#         if h > w:
#             pad_w = (h-w) // 2
#             img = np.concatenate((np.zeros((h, pad_w, 3)), img), axis=1)
#             img = np.concatenate((img, np.zeros((h, pad_w, 3))), axis=1)
#         elif h < w:
#             pad_h = (w - h) // 2
#             img = np.concatenate((np.zeros((pad_h, w, 3)), img), axis=0)
#             img = np.concatenate((img, np.zeros((pad_h, w, 3))), axis=0)
#
#         if self.transform:
#             img = Image.fromarray(img.astype(np.uint8))
#             img = self.transform(img)
#         else:
#             img = cv2.resize(img, (224, 224))
#             img = img/255.
#             img = img.transpose((2, 0, 1))
#             img = torch.from_numpy(img)
#         return img, label
#
#     def __len__(self):
#         return len(self.img_files)


def get_normal(files):
    data = np.empty(0)
    for file in files:
        data = np.append(data, np.loadtxt(file))
    normal_param = {}
    normal_param['min'] = np.min(data)
    normal_param['max'] = np.max(data)
    return normal_param

def gen_dataset(files, stride, norm_dict, seq_len, out_len):
    data = np.empty((0, seq_len, 1))
    label = np.empty((0, out_len))
    if isinstance(files, list):
        for file in tqdm(files):
            sequence = np.loadtxt(file)
            sequence = (sequence - norm_dict['min']) / (norm_dict['max'] - norm_dict['min'])
            for i in range(0, sequence.shape[0], stride):
                if i + seq_len + out_len <= sequence.shape[0]:
                    data = np.concatenate((data, sequence[i:i+seq_len].reshape((1, seq_len, 1))), axis=0)
                    label = np.concatenate((label, sequence[i+seq_len:i+seq_len+out_len].reshape((1, -1))), axis=0)
    else:
        sequence = np.loadtxt(files)
        sequence = (sequence - norm_dict['min']) / (norm_dict['max'] - norm_dict['min'])
        for i in range(0, sequence.shape[0], stride):
            if i + seq_len + out_len <= sequence.shape[0]:
                data = np.concatenate((data, sequence[i:i + seq_len].reshape((1, seq_len, 1))), axis=0)
                label = np.concatenate((label, sequence[i + seq_len:i + seq_len + out_len].reshape((1, -1))), axis=0)
    return data, label
