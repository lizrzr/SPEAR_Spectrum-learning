import argparse
import os
import re
import glob
import scipy.io as sio
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import modelSPEAR
from datasets import trainset_loader
from datasets import testset_loader
from torch.utils.data import DataLoader
import dct
import torchinfo
# 引入这一块是为了更好的调参
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=40, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=2.5e-4, help="adam: learning rate")
parser.add_argument("--n_cpu", type=int, default=5)
parser.add_argument("--model_save_path", type=str, default="saved_FBPConvNetmodels/3th")
parser.add_argument('--checkpoint_interval', type=int, default=1)

opt = parser.parse_args()
cuda = True if torch.cuda.is_available() else False
imagesize = 256


class net():
    def __init__(self):
        # 网络模型为modelFBPConvNet
        self.model = modelSPEAR.FBPConvNet()
        # 损失函数计算方法为平方根
        self.loss = nn.MSELoss()
        #
        self.path = opt.model_save_path
        # 利用Dataloader装载训练数据
        self.train_data = DataLoader(trainset_loader(),
                                     batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)
        # 利用Dataloader装载测试数据
        self.test_data = DataLoader(testset_loader(),
                                    batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
        # optimizer优化器，第一个self.model.parameters()是必备的，lr代表学习速率
        self.optimizer = optim.Adam(self.model.parameters(), lr=opt.lr)
        self.start = 0
        self.epoch = opt.epochs
        self.check_saved_model()
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=5, gamma=0.8)
        if cuda:
            self.model = self.model.cuda()

    def check_saved_model(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            self.initialize_weights()
        else:
            model_list = glob.glob(self.path + '/FBPConvNetmodel_epoch_*.pth')
            if len(model_list) == 0:
                self.initialize_weights()
            else:
                last_epoch = 0
                for model in model_list:
                    epoch_num = int(re.findall(r'FBPConvNetmodel_epoch_(-?[0-9]\d*).pth', model)[0])
                    if epoch_num > last_epoch:
                        last_epoch = epoch_num
                self.start = last_epoch
                self.model.load_state_dict(torch.load(
                    '%s/FBPConvNetmodel_epoch_%04d.pth' % (self.path, last_epoch)))

    def displaywin(self, img):
        img[img < 0] = 0
        high = img.max()
        low = 0
        img = (img - low) / (high - low) * 255
        return img

    def initialize_weights(self):
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0, std=0.01)
                if module.bias is not None:
                    module.bias.data.zero_()

            if isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def train(self):
        pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Total_params: {}".format(pytorch_total_params))
        for epoch in range(self.start, self.epoch):
            for batch_index, data in enumerate(self.train_data):
                input_data, label_data = data
                if cuda:
                    input_data = input_data.cuda()
                    label_data = label_data.cuda()
                # 下面五行为训练的核心行为
                # 1.optimizer.zero_grad优化器梯度清零，目的是防止上次的梯度干扰
                # 2.输入经过网络计算输出
                # 3.计算损失函数
                # 4.反向传播
                # 5.优化器优化
                input_data = F.interpolate(label_data, 256, mode='bicubic', align_corners=True)
                self.optimizer.zero_grad()
                output = self.model(input_data)
                outputs = dct.tensor_dct2D(output, 4)
                label_datas = dct.tensor_dct2D(label_data, 4)
                loss = 0.8 * self.loss(output, label_data) + 0.2 * self.loss(outputs, label_datas)
                loss.backward()
                self.optimizer.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d]: [loss: %f]"
                    % (epoch + 1, self.epoch, batch_index + 1, len(self.train_data), loss.item())
                )
            self.scheduler.step()
            if opt.checkpoint_interval != -1 and (epoch + 1) % opt.checkpoint_interval == 0:
                torch.save(self.model.state_dict(), '%s/FBPConvNetmodel_epoch_%04d.pth' % (self.path, epoch + 1))

    def test(self):
        losstest = 0
        count = 0
        for batch_index, data in enumerate(self.test_data):
            input_data, label_data, res_name = data
            if cuda:
                input_data = input_data.cuda()
                label_data = label_data.cuda()
                input_data = F.interpolate(label_data, 256, mode='bicubic', align_corners=True)
            with torch.no_grad():
                output = self.model(input_data)
            res = output.cpu().numpy()
            reference = label_data.cpu().numpy()
            inputs = input_data.cpu().numpy()
            losstest = losstest + torch.sum(torch.square(label_data - output) / imagesize / imagesize)

            for i in range(output.shape[0]):
                sio.savemat(res_name[i], {'data': res[i, 0], 'reference': reference[i, 0], 'inputs': inputs[i, 0]})

        print(" [loss: %f]" % (losstest))


if __name__ == "__main__":
    network = net()
    #查看网络参数量
    pytorch_total_params = sum(p.numel() for p in network.model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))
    #network.train()
    #network.test()
