import glob
import scipy.io as scio
import torch
from torch.utils.data import Dataset


class trainset_loader(Dataset):
    def __init__(self):
        # glob.glob()函数，将目录下所有跟通配符模式相同的文件放到一个列表中。
        self.files_A = sorted(glob.glob('.././train/' + '*.mat'))

    def __getitem__(self, index):
        file_A = self.files_A[index]
        file_B = file_A
        # 读取.mat，原本读完是dic字典型，【‘data’】转换为了narray
        label_data = scio.loadmat(file_B)['label']
        input_data = scio.loadmat(file_B)['data']
        label_data = label_data / label_data.max()
        input_data = torch.FloatTensor(input_data).unsqueeze_(0)
        label_data = torch.FloatTensor(label_data).unsqueeze_(0)
        return input_data, label_data

    def __len__(self):
        return len(self.files_A)


class testset_loader(Dataset):
    def __init__(self):
        self.files_A = sorted(glob.glob('.././test/' + '*.mat'))

    def __getitem__(self, index):
        file_A = self.files_A[index]
        file_B = file_A
        res_name = './/result//' + file_A[-11:]
        input_data = scio.loadmat(file_B)['data']
        label_data = scio.loadmat(file_B)['label']
        label_data = label_data / label_data.max()
        input_data = torch.FloatTensor(input_data).unsqueeze_(0)
        label_data = torch.FloatTensor(label_data).unsqueeze_(0)

        return input_data, label_data, res_name

    def __len__(self):
        return len(self.files_A)
