import argparse
import os
import datetime

import sys
import matplotlib.pyplot as plt
import torch.utils.tensorboard as tb
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from Dataset1 import ClassDataset
from model import Model as Model
import torch.optim as optim
import time
import torch.nn as nn
from loss import DisLoss, LargeMarginLoss
from plot_image import classification, padding_image
from sklearn.decomposition import PCA

plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def set_init():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument('-config', help='Set configs file', default="cfg.yml", type=str)
    parser.add_argument("--resume_training", action="store_true", help="Whether to resume training", default=False)
    parser.add_argument("--seed", help="Whether to resume training", default=10, type=int)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    try:
        os.mkdir('tensorboard')
        os.mkdir('checkpoint')
    except:
        pass
    tb_path = f"tensorboard/tb_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = f"checkpoint/ckpt_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    args.checkpoint_dir = checkpoint_dir
    tb_logger = tb.SummaryWriter(log_dir=tb_path)

    # 获取当前日期和时间
    now = datetime.datetime.now()

    # 格式化时间输出
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
    #  ---  Set random seed   ---  #
    try:
        with open('../seed/logs.txt', 'a') as f:
            f.write(f'{formatted_time}\t使用随机种子：{args.seed}')
        torch.manual_seed(args.seed)
    except:
        pass
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    return args, config, tb_logger


class DoGL1Loss(nn.Module):
    def __init__(self, level=4):
        super().__init__()
        self.level = level
        self.l1Loss = nn.L1Loss()

    def forward(self, x, gt):
        b, n, h, w = x.shape
        losses = []
        for i in range(n):
            xx = x[:, i, :, :]
            gg = gt[:, i, :, :]
            losses.append(self.l1Loss(xx, gg))
        return losses


def get_Loss(x: torch.Tensor, gt: torch.Tensor, feature:torch.Tensor, alpha=0, m=3):
    assert x.shape[0] == gt.shape[0], f"x and gt not match, x:{x.shape[0]}, gt:{gt.shape[0]}"
    ce = nn.CrossEntropyLoss()
    # x = torch.argmax(x, 1)
    loss1 = ce(x, gt)
    if alpha:
        add1loss = LargeMarginLoss(m=m)
        loss2 = add1loss(feature)
    else:
        loss2 = 0
    return loss1, loss2, loss1 + alpha * loss2

class Run():
    def __init__(self, datasetname='indian pines', model='transformer'):
        self.datasetname = datasetname
        self.args, self.config, self.tb_logger = set_init()
        os.mkdir(os.path.join(self.args.checkpoint_dir, 'model'))
        os.mkdir(os.path.join(self.args.checkpoint_dir, 'image'))
        os.mkdir(os.path.join(self.args.checkpoint_dir, 'image', 'train'))
        os.mkdir(os.path.join(self.args.checkpoint_dir, 'image', 'valid'))
        os.mkdir(os.path.join(self.args.checkpoint_dir, 'image', 'test'))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f'device: {self.device}')
        datasetname = datasetname.lower()
        self.now_dataset = self.config.dataset.ip if datasetname == 'indian pines'\
            else (self.config.dataset.pu if datasetname == 'pavia university'
                  else (self.config.dataset.ksc if datasetname == 'ksc'
                        else (self.config.dataset.pc if datasetname == 'pavia center'
                              else (self.config.dataset.botswana if datasetname == 'botswana'
                                    else (self.config.dataset.salinas if datasetname == 'salinas' else (
                                        self.config.dataset.hanchuan if datasetname == 'hanchuan' else (
                                            self.config.dataset.longkou if datasetname == 'longkou' else None)))))))
        self.class_num = self.now_dataset.class_num
        print(f'model: {model}')
        if model[:3] == 'res':
            self.model = ResModel(
                self.config,
                self.now_dataset
            ).to(self.device)
        else:
            self.model = Model(
                self.config,
                self.now_dataset,
                self.now_dataset.pca_channel if self.now_dataset.pca else self.now_dataset.in_channel,
                self.config.data.input_size,
                self.now_dataset.class_num,
                split_class=self.now_dataset.class_num if self.config.model.split_class else 0,
                device=self.device
            ).to(self.device)
        if self.config.train.parrllel:
            self.model = torch.nn.DataParallel(self.model, device_ids=[1])
        # inputs1 = torch.randn(1, 3, 16, 16).cuda()
        # inputs2 = torch.randn(1, 6, 8, 8).cuda()
        # # e = torch.from_numpy(np.array(pywt.wavedec2(inputs1.cpu(), wavelet='db1', level=1)[0], dtype=np.float32)).cuda()
        # flops, params = profile(self.model, (inputs1, inputs2))
        # print('flops: ', flops, 'params: ', params)

    def train(self, datasetname, log=False, lr=None):
        if lr is None:
            lr = self.config.optim.lr
        best_aa = 0
        print('使用学习率：', lr)
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, betas=(0.9, 0.99),
                                weight_decay=self.config.optim.weight_decay)
        # scheduler = LinearLR(optimizer, self.config.optim.lr, self.config.optim.min_lr, self.config.train.total_iters)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config.train.epochs//3, eta_min=lr/500)

        current_iter = 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.config.data.pre_model, "gopro_ckpt.pth"))
            self.model.load_state_dict(states)

        dataset = ClassDataset(self.datasetname, self.config, self.now_dataset, self.now_dataset.class_num, self.now_dataset.per_class_num, mode='train')
        train_loader = DataLoader(
            dataset,
            batch_size=self.config.train.batch_size,
            shuffle=True,
            num_workers=8
        )
        print('-------------start----------------')
        if os.path.exists('./logg'):
            os.mkdir('./logg')
        with open(f'./logg/logging_{datasetname}_{lr}{"-pca" if self.now_dataset.pca else ""}.txt', 'a') as f:
            f.write(f'----------------------start: {self.args.checkpoint_dir} --------------------------\nmin_data: {dataset.min_data}, max_data: {dataset.max_data}\n')
        epoch = 0
        for i in range(len(dataset.class_origin_num)):
            print(f'class-{i} origin num: {dataset.class_origin_num[i]}')
            with open(f'./logg/logging_{datasetname}_{lr}{"-pca" if self.now_dataset.pca else ""}.txt', 'a') as f:
                f.write(f'class-{i} origin num: {dataset.class_origin_num[i]}\n')
        while epoch < self.config.train.epochs:
            train_matrix = torch.zeros((self.class_num, self.class_num))
            train_true_count, train_all_count = 0, 0
            train_loss_list, valid_loss_list = [], []
            train_loss1_list, train_loss2_list = [], []
            # while current_iter <= self.config.train.total_iters:
            #     if current_iter > self.config.train.total_iters:
            #         break
            for batch_idx, (data, wvt, target) in enumerate(train_loader):
                # print(data.shape, wvt.shape, target.shape)
                # current_iter += 1
                self.model.train()

                data = data.to(self.device)
                wvt = wvt.to(self.device)
                target = target.to(self.device)

                # net.eval()
                # net2.train()
                # data2 = net1(data)
                # out = net2(data2)

                optimizer.zero_grad()
                data_start = time.time()
                ifn = self.model(data)
                one_iter_time = time.time() - data_start

                # loss, losses, weights = get_Loss(ifn, target)
                loss1, loss2, loss = get_Loss(ifn.clone(), target, ifn)
                train_loss1_list.append(loss1.item())
                if not isinstance(loss2, int):
                    train_loss2_list.append(loss2.item())
                else:
                    train_loss2_list.append(loss2)
                train_loss_list.append(loss.item())
                train_true_count += self.accuracy(ifn, target, train_matrix)
                train_all_count += ifn.shape[0]

                loss.backward()
                optimizer.step()
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        if log:
                            print(f"{name} 梯度: {param.grad}")
                        if self.config.train.grad_clip:
                            param.grad.data.clamp_(min=self.config.train.min_grad, max=self.config.train.max_grad)

                if batch_idx % 100 == 0:
                    train_loss = sum(train_loss_list) / len(train_loss_list)
                    train_oa = train_true_count / train_all_count
                    kappa = self.kappa(train_matrix, train_oa)
                    aa = self.AA(train_matrix)
                    lr_num = optimizer.state_dict()['param_groups'][0]['lr']
                    print(f'epoch-{epoch + 1}:    {batch_idx} / {len(train_loader)}   loss:{train_loss}   train_oa: {train_oa}   '
                          f'kappa: {kappa}    AA:{aa}    one_iter_time:{one_iter_time}    lr:{lr_num}')
                    with open(f'./logg/logging_{datasetname}_{lr}{"-pca" if self.now_dataset.pca else ""}.txt', 'a') as f:
                        f.write(f'epoch-{epoch + 1}:    {batch_idx} / {len(train_loader)}   loss:{train_loss}   train_oa: {train_oa}   ')
                        f.write(f'kappa: {kappa}    AA:{aa}    one_iter_time:{one_iter_time}    lr:{lr_num}\n')

            scheduler.step()
            if (epoch + 1) % 5 == 0:
                self.save_model(datasetname, lr)
            aa = self.val(f'./logg/logging_{datasetname}_{lr}{"-pca" if self.now_dataset.pca else ""}.txt', f'{self.args.checkpoint_dir}/image/valid/训练混淆矩阵_{datasetname}_{epoch}_{lr}.jpg', 'valid', epoch)
            if aa > best_aa:
                self.save_model(datasetname, lr, os.path.join(self.args.checkpoint_dir, 'model', f"best_model_{datasetname}_{lr}.pth"))
                best_aa = aa
            train_loss = sum(train_loss_list) / len(train_loss_list)
            train_loss1 = sum(train_loss1_list) / len(train_loss1_list)
            train_loss2 = sum(train_loss2_list) / len(train_loss2_list)
            train_oa = train_true_count / train_all_count
            kappa = self.kappa(train_matrix, train_oa)
            aa = self.AA(train_matrix)
            self.tb_logger.add_scalar("train_loss1", train_loss1, global_step=epoch)
            self.tb_logger.add_scalar("train_loss2", train_loss2, global_step=epoch)
            self.tb_logger.add_scalar("train_loss", train_loss, global_step=epoch)
            self.tb_logger.add_scalar("train_OA", train_oa, global_step=epoch)
            self.tb_logger.add_scalar("train_Kappa", kappa, global_step=epoch)
            self.tb_logger.add_scalar("train_AA", aa, global_step=epoch)

            print(f'epoch-{epoch + 1}/{self.config.train.epochs}    loss:{train_loss}   train_oa: {train_oa}   '
                  f'kappa: {kappa}    AA:{aa}')
            with open(f'./logg/logging_{datasetname}_{lr}{"-pca" if self.now_dataset.pca else ""}.txt', 'a') as f:
                f.write(f'epoch-{epoch + 1}/{self.config.train.epochs}    loss: {train_loss}   train_oa: {train_oa}   ')
                f.write(f'kappa: {kappa}    AA: {aa}\n')
            epoch += 1
            self.log(train_matrix, 'train', f'./logg/logging_{datasetname}_{lr}{"-pca" if self.now_dataset.pca else ""}.txt')
            if (epoch + 1) % 10 == 0 or self.config.train.epochs == epoch + 1:
                self.plot_matrix(
                    train_matrix / (torch.sum(train_matrix, 0).unsqueeze(0) +1e-10),
                    f'{self.args.checkpoint_dir}/image/train/训练混淆矩阵_{datasetname}_{epoch}_{lr}.jpg'
                )
        self.save_model(datasetname, lr)
        print(f'best valid AA: {best_aa}')
        print(' --- last test ---\n')
        with open(f'./logg/logging_{datasetname}_{lr}{"-pca" if self.now_dataset.pca else ""}.txt', 'a') as f:
            f.write(' --- last test ---\n')
        self.val(f'./logg/logging_{datasetname}_{lr}{"-pca" if self.now_dataset.pca else ""}.txt',
                 f'{self.args.checkpoint_dir}/image/test/last_训练混淆矩阵_{datasetname}_{epoch}_{lr}.jpg',
                 'test')
        # 测试最优模型
        states = torch.load(os.path.join(self.args.checkpoint_dir, 'model', f"best_model_{datasetname}_{lr}.pth"))
        self.model.load_state_dict(states[0])
        print(' --- best test ---\n')
        with open(f'./logg/logging_{datasetname}_{lr}{"-pca" if self.now_dataset.pca else ""}.txt', 'a') as f:
            f.write(' --- bests test ---\n')
        self.val(f'./logg/logging_{datasetname}_{lr}{"-pca" if self.now_dataset.pca else ""}.txt',
                 f'{self.args.checkpoint_dir}/image/test/best_训练混淆矩阵_{datasetname}_{epoch}_{lr}.jpg',
                 'test')

    def save_model(self, datasetname, lr, name=None):
        states = self.model.state_dict(),
        if name is None:
            torch.save(states, os.path.join(self.args.checkpoint_dir, 'model', f"class_ckpt_{datasetname}_{lr}.pth"))
        else:
            torch.save(states, name)
        print(f'model saved')
    def val(self, filename, picname, mode='val', epoch=0):
        with open(filename, 'a') as f:
            f.write(f'{mode}...\n')
        val_dataset = ClassDataset(self.datasetname, self.config, self.now_dataset, self.now_dataset.class_num, self.now_dataset.per_class_num, mode=mode)
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=False,
            num_workers=8
        )
        val_true_count, val_all_count = 0, 0
        val_time_list = []
        self.model.eval()
        valid_matrix = torch.zeros((self.class_num, self.class_num))
        with torch.no_grad():
            for idx, (data, event, target) in enumerate(val_loader):
                data, target, event = data.to(self.device), target.to(self.device), event.to(self.device)
                btime = time.time()
                ifn = self.model(data)
                val_true_count += self.accuracy(ifn, target, valid_matrix)
                val_all_count += ifn.shape[0]
                dtime = time.time() - btime
                val_time_list.append(dtime)
            val_mean_oa = val_true_count / val_all_count
            val_mean_time = sum(val_time_list) / len(val_time_list)

            self.tb_logger.add_scalar(f"{mode}OA", val_mean_oa, global_step=epoch)
            self.tb_logger.add_scalar(f"{mode}AA", self.AA(valid_matrix), global_step=epoch)
            self.tb_logger.add_scalar(f"{mode}kappa", self.kappa(valid_matrix, val_mean_oa), global_step=epoch)
            print('epoch-', epoch + 1, ': ')
            print(f'{mode} OA: {val_mean_oa}')
            print(f'{mode} AA: {self.AA(valid_matrix)}')
            print(f'{mode} kappa: {self.kappa(valid_matrix, val_mean_oa)}')
            print(f'{mode} average time: {val_mean_time}')
            with open(filename, 'a') as f:
                f.write(f'{epoch}: \n')
                f.write(f'{mode} OA: {val_mean_oa}\n')
                f.write(f'{mode} AA: {self.AA(valid_matrix)}\n')
                f.write(f'{mode} kappa: {self.kappa(valid_matrix, val_mean_oa)}\n')
        self.log(valid_matrix, mode, filename)
        if mode == 'val':
            if (epoch + 1) % 10 == 0 or self.config.train.epochs == epoch + 1:
                self.plot_matrix(valid_matrix / (torch.sum(valid_matrix, 0).unsqueeze(0) +1e-10), picname)
        else:
            self.plot_matrix(valid_matrix / (torch.sum(valid_matrix, 0).unsqueeze(0) + 1e-10), picname)
        return self.AA(valid_matrix)
    def log(self, matrix, mode, path=''):
        out = matrix / (torch.sum(matrix, 0).unsqueeze(0) + 1e-10)
        print(f'------------{mode}----------------')
        if len(path) > 0:
            with open(path, 'a') as f:
                f.write(f'------------{mode}----------------\n')
        for i in range(out.shape[0]):
            print(f'第{i}类预测正确个数： {int(out[i, i] * 10000) / 100}%')
            if len(path) > 0:
                with open(path, 'a') as f:
                    f.write(f'第{i}类预测正确个数： {int(out[i, i] * 10000) / 100}%\n')
    def plot_matrix(self, matrix, filename, title='confusion matrix'):
        matrix = matrix.T #实际使用与定义相反
        classes = self.now_dataset.classes
        fig, ax = plt.subplots(figsize=(self.now_dataset.plot_size, self.now_dataset.plot_size))
# cfg添加绘画大小 本文件添加plt中文正确显示 绘制设置
        plt.rcParams['font.size'] = self.now_dataset.font_size
        plt.rcParams['axes.titlesize'] = self.now_dataset.title_size
        plt.rcParams['axes.labelsize'] = self.now_dataset.axes_size
        plt.rcParams['xtick.labelsize'] = self.now_dataset.tick_size
        plt.rcParams['ytick.labelsize'] = self.now_dataset.tick_size
        plt.rcParams['legend.fontsize'] = self.now_dataset.legend_size
        imshow_temp = ax.imshow(matrix, cmap='magma')
        plt.colorbar(imshow_temp)
        # 关闭轴的坐标轴
        ax.set_xticks([])
        ax.set_yticks([])
        # 在矩阵的特定位置写入文本
        plt.title(title)  # 改图名
        if len(classes) == matrix.shape[0]:
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=-45)
            plt.yticks(tick_marks, classes)
        #错误
        #plt.xlabel('Actual Category')
        #plt.ylabel('Prediction Category')
        plt.xlabel('Prediction Category')
        plt.ylabel('Truth Category')
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                text = f"{int(matrix[i, j] * 10000) / 100}%"
                ax.text(j, i, text, va='center', ha='center', color='white' if matrix[i, j] < 0.7 else 'black', fontsize=14)
                # ax.text(j, i, text, va='center', ha='center', color='white')
        plt.savefig(filename, bbox_inches='tight')
    def kappa(self, matrix, oa):
        a, b = torch.sum(matrix, 1), torch.sum(matrix, 0)
        pe = torch.sum(a * b) / torch.sum(a) / torch.sum(b)
        return (oa - pe) / (1 - pe)

    def AA(self, matrix):
        out = matrix / (torch.sum(matrix, 0).unsqueeze(0) +1e-10)
        return torch.trace(out) / out.shape[-1]

    def accuracy(self, pre, label, matrix):
        x1 = torch.argmax(pre, dim=1)
        # x2 = torch.argmax(label, dim=1)
        x2 = label
        for i in range(label.shape[0]):
            matrix[x1[i], x2[i]] += 1
        return (x1 == x2).sum()
    def _classification(self, net_path, data_path, img_size, batch_size, save_path, log):
        states = torch.load(net_path)
        self.model.load_state_dict(states[0])
        img = torch.load(data_path).permute(2, 0, 1)
        if self.now_dataset.pca:
            pca = PCA(n_components=64)
            data_np = img.numpy()
            data_pca = pca.fit_transform(data_np.reshape(-1, data_np.shape[-1])).reshape(data_np.shape[0],
                                                                                         data_np.shape[1], -1)
            img = torch.from_numpy(data_pca)
        img = padding_image(img, img_size)
        max_data, min_data = torch.max(img), torch.min(img)
        with open(log, 'a') as f:
            f.write(f'file name: {data_path}\nmin_data: {min_data}, max_data: {max_data}\n')
        im = (img - min_data) / (max_data - min_data)
        self.model.eval()
        classification(im, self.model, batch_size, img_size, log, save_path=save_path, device=self.device)
    def _concat_result(self, pre_path, last_path, plot):
        content = []
        i = 0
        try:
            while os.path.exists(pre_path + str(i) + last_path):
                content.append(torch.load(pre_path + str(i) + last_path))
                i += 1
            if i > 0:
                content = torch.concat(content, dim=-1)
                torch.save(content, pre_path + last_path)
                if plot:
                    plt.imshow(content)
                    plt.savefig(pre_path + '.png')
                    plt.close()
                i -= 1
                while i >= 0:
                    os.remove(pre_path + str(i) + last_path)
                    i -= 1
        except:
            pass

    def classification(self, net_path, data_path, img_size, batch_size, save_path, log):
        self._classification(net_path, data_path, img_size, batch_size, save_path, log)
        self._concat_result(save_path[:-4] + '_result', '.pt', True)
        self._concat_result(save_path[:-4] + '_pres', '.pt', False)

if __name__ == '__main__':
    # 检查是否传入了命令行参数
     if len(sys.argv) > 2 and sys.argv[1] == '--seed':
         seed = int(sys.argv[2])  # 获取第一个参数
     else:
         seed = 10

     datasetname = 'longkou'
     for lr in [1e-3, 5e-4, 1e-4, 5e-5]:
         print('datasetname: ', datasetname)
         r = Run(datasetname=datasetname)
         r.train(datasetname, lr=lr)
     
     with open(f'../random/{seed}.txt', 'a') as f:
         for i in np.random.rand(10):
             f.write(f'{i} ')
         f.write(f'\n')
    
