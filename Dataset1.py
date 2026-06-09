import os
import pywt, torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms as T
class ClassDataset(Dataset):
    def __init__(self, datasetname, config, now_dataset, class_num, per_class_num, mode='train', wavelet='db1', level=1, train=7, valid=2, test=1):
        super().__init__()
        self.level = level
        self.config = config
        self.wavelet = wavelet
        self.now_dataset = now_dataset
        self.class_origin_num = [] #打印数据增广前数据各类别数量
        if mode == 'train':
            self.datasets, self.targets = self.read_file_to_n(datasetname, class_num, per_class_num, 'train', argument_mode=config.data.argument_mode)
        elif mode == 'valid':
            self.datasets, self.targets = self.read_file_mul_n(datasetname, class_num, per_class_num, 'valid')
        elif mode == 'test':
            self.datasets, self.targets = self.read_file_mul_n(datasetname, class_num, per_class_num, 'test')
        print(mode, self.datasets.shape, self.targets.shape)
    def padding_image(self, data, size):
        size1 = size[0] // 2
        c, h, w = data.shape
        data = torch.concat([torch.zeros((c, size1, w)), data, torch.zeros((c, size[0]-size1, w))], dim=1)
        h += size[0]
        size1 = size[1] // 2
        data = torch.concat([torch.zeros((c, h, size1)), data, torch.zeros((c, h, size[1]-size1))], dim=2)
        return data
    def read_file_to_n(self, datasetname, class_num, per_class_num, mode='', argument_mode='salt_noise'):
        if mode == 'train' and per_class_num != -1:
            print(f'使用数据增广:{argument_mode}')
        img, target = [[] for _ in range(class_num)], []
        img_size = self.config.data.input_size
        part_size = (img_size[0] // 2, img_size[1] // 2)
        path = os.path.join(self.config.data.main_path, datasetname)
        data = torch.load(os.path.join(path, 'data-pca.pkl' if self.now_dataset.pca else 'data.pkl')).permute(2, 0, 1)
        #with open('./error/log.txt', 'a') as f:
         #   f.write(f'{data.shape}\n')
        data = self.padding_image(data, self.config.data.input_size)
        max_data, min_data = torch.max(data), torch.min(data)
        self.max_data, self.min_data = max_data, min_data
        # temp = data.reshape(-1)
        # data /= torch.sort(temp)[0][temp.size()[0] * 95 // 100]

        # data = (data - torch.min(data)) / (torch.max(data) - torch.min(data))
        # print(data)
        label = torch.load(os.path.join(path, 'label.pkl')).long()
        #with open('./error/log.txt', 'a') as f:
         #   f.write(f'{label.shape}\n')
        with open(os.path.join(path, f'{mode}.txt'), 'r') as f:
            contents = f.read().split('|')[:-1]
        size1 = label.shape[-1]
        for point in contents:
            x, y = int(point) // size1, int(point) % size1
            try:
                #temp_img = data[:, x - part_size[0]:x + (img_size[0] - part_size[0]), y - part_size[1]:y + (img_size[1] - part_size[1])]
                temp_img = data[:, x:x + img_size[0], y:y + img_size[1]]
                if temp_img.shape[1] == img_size[0] and temp_img.shape[2] == img_size[1]:
                    img[label[x, y]-1].append(temp_img)
            except:
                pass
        for i in range(class_num):
            temp = []
            if mode == 'train':
                if len(self.class_origin_num) == i:
                    self.class_origin_num.append(len(img[i]))
          #          with open('./error/log.txt', 'a') as f:
           #             f.write(f'class-{i} num: {self.class_origin_num[-1]}\n')
                if per_class_num != -1:
                    while len(img[i]) + len(temp) < per_class_num:
                        if argument_mode == 'salt_noise':
                            temp.append(self.data_argument_salt_noise(img[i][len(temp) % len(img[i])].clone(), len(temp) // len(img[i])))
                        else:
                            temp.append(self.data_argument_flip(img[i][len(temp) % len(img[i])].clone(), len(temp) // len(img[i])))
                    img[i].extend(temp)
                    if len(img[i]) > per_class_num:
                        img[i] = img[i][:per_class_num]
            target.append(torch.ones((len(img[i]))).long() * i)
        imgs = []
        for i in range(len(img)):
            imgs.extend(img[i])
        imgs = torch.stack(imgs, dim=0)
        return (imgs - min_data) / (max_data - min_data), torch.concat(target, dim=0)


    def read_file_mul_n(self, datasetname, class_num, per_class_num, mode=''):
        if mode == 'train' and per_class_num != -1:
            print('使用数据增广')
        img, target = [], []
        img_size = self.config.data.input_size
        part_size = (img_size[0] // 2, img_size[1] // 2)
        path = os.path.join(self.config.data.main_path, datasetname)
        data = torch.load(os.path.join(path, 'data-pca.pkl' if self.now_dataset.pca else 'data.pkl')).permute(2, 0, 1)
        data = self.padding_image(data, self.config.data.input_size)
        max_data, min_data = torch.max(data), torch.min(data)
        self.max_data, self.min_data = max_data, min_data
        # temp = data.reshape(-1)
        # data /= torch.sort(temp)[0][temp.size()[0] * 95 // 100]

        # data = (data - torch.min(data)) / (torch.max(data) - torch.min(data))
        # print(data)
        label = torch.load(os.path.join(path, 'label.pkl')).long()
        with open(os.path.join(path, f'{mode}.txt'), 'r') as f:
            contents = f.read().split('|')[:-1]
        print(len(contents))
        size1 = label.shape[-1]
        for point in contents:
            x, y = int(point) // size1, int(point) % size1
            #temp_img = data[:, x - part_size[0]:x + (img_size[0] - part_size[0]),
             #          y - part_size[1]:y + (img_size[1] - part_size[1])]
            temp_img = data[:, x:x + img_size[0], y:y + img_size[1]]
            if temp_img.shape[1] == img_size[0] and temp_img.shape[2] == img_size[1]:
                img.append(temp_img)
                target.append(label[x, y] - 1)

        if mode == 'train':
            imgs = [data for data in img]
            targets = [data for data in target]
            # 水平翻转的概率为0.8
            h_flip = transforms.RandomHorizontalFlip(p=0.8)
            # 垂直翻转的概率为0.8
            v_flip = transforms.RandomVerticalFlip(p=0.8)
            rotate90 = T.RandomRotation(90)
            rotate180 = T.RandomRotation(180)
            rotate270 = T.RandomRotation(270)
            for i, temp_img in enumerate(img):
                targets.extend([target[i] for _ in range(min(self.config.data.rotate.num, 5))])
                if self.config.data.rotate.num > 0:
                    imgs.append(h_flip(temp_img))
                if self.config.data.rotate.num > 1:
                    imgs.append(v_flip(temp_img))
                if self.config.data.rotate.num > 2:
                    imgs.append(rotate90(temp_img))
                if self.config.data.rotate.num > 3:
                    imgs.append(rotate180(temp_img))
                if self.config.data.rotate.num > 4:
                    imgs.append(rotate270(temp_img))
                if i % 1000 == 0:
                    print(f'{self.config.data.rotate.num}倍训练集生成--   {int((i + 1) / len(img) * 10000) / 100}%  ( {i + 1}  /  {len(img)} )   --')
            img = imgs
            target = targets
        print(min_data, max_data)
        return (torch.stack(img, dim=0) - min_data) / (max_data - min_data), torch.stack(target, dim=0)

    def data_argument_salt_noise(self, img, ind, probability=0.1, salt=False): # probability是噪声的概率
        try:
            if ind == 0:
                # 调整亮度
                return transforms.ColorJitter(brightness=0.5,contrast=0,saturation=0,hue=0)(img)
            elif ind == 1:
                # 调整对比度
                return transforms.ColorJitter(brightness=0,contrast=0.5,saturation=0,hue=0)(img)
            elif ind == 2:
                # 调整饱和度
                return transforms.ColorJitter(brightness=0,contrast=0,saturation=0.5,hue=0)(img)
        except:
            pass
        if salt:
            mode = 2
        else:
            mode = 1
        if ind % mode:
            # 添加高斯噪声
            loc = torch.tensor([0.0])  # 均值
            scale = torch.tensor([1.0])  # 标准差
            return img + torch.distributions.Normal(loc, scale).sample(img.shape).squeeze(-1)

        # 添加椒盐噪声
        channels, height, width = img.size()
        # 创建一个与输入Tensor相同大小的布尔掩码，用于选择需要添加噪声的位置
        noise_mask = torch.zeros_like(img, dtype=torch.bool)
        # 随机设置一些位置为True，代表需要添加噪声的位置
        noise_mask[:channels, :height, :width] = torch.bernoulli(
            torch.full((channels, height, width), probability)).bool()
        # 在选定的位置上，随机设置为1（椒盐噪声）或0（未改变）
        data1,data2 = torch.max(img), torch.min(img)
        data1 = data1 - data2
        img[noise_mask] = torch.randint(0, 2, size=(noise_mask.sum(),), dtype=img.dtype, device=img.device) * data1 + data2
        return img

    def data_argument_flip(self, img, ind, p=0.8, max_size=3, max_num=4, max_noise=10, noise_p=0.1, save_mid=True): # probability是翻转概率
        mod = ind % 9
        if mod < 2:
            return img
        if mod == 3:
            if ind // 9 % 2 == 0:
                h_flip = transforms.RandomHorizontalFlip(p=p)# 水平翻转
                return h_flip(img)
            v_flip = transforms.RandomVerticalFlip(p=p)# 垂直翻转
            return v_flip(img)
        if mod == 4:
            if ind // 9 % 2 == 0:
                return T.RandomRotation(90)(img)
            return T.RandomRotation(180)(img)
        if mod == 5:
            return T.RandomRotation(270)(img)
        if mod == 6:
            size1, size2 = img.shape[1:]
            msk = self.generate_mask(size1, size2, max_num, max_size, save_mid=save_mid)
            # print(msk.sum())
            img[:, msk] = 0
        elif mod == 7:
            noise = torch.randn(img.shape)
            # 生成一个随机的掩码 (mask)，其中一定比例的元素为 0，其余为 1
            mask = torch.rand(img.shape) > noise_p  # 随机生成一个与矩阵大小相同的掩码，比例为 noise_p
            noise[mask] = 0
            noise[noise > max_noise] = max_noise
            noise[noise < -max_noise] = -max_noise
            # print((mask != True).sum())
            img += noise
        return img

    def generate_mask(self, size1, size2, max_num, max_size, save_mid=True):
        matrix = torch.zeros((size1, size2), dtype=torch.bool)
        def modify_matrix(matrix, m, n, k, max_size, save_mid=True):
            if k == 0:
                return matrix
            # 随机选择k个位置
            positions = np.random.choice(np.arange(m * n), k, replace=False)

            # 对每个位置修改i*i大小区域的值
            for pos in positions:
                x, y = pos // size2, pos % size2
                # 确保(i*i)区域不会超出矩阵边界
                i = np.random.randint(0, max_size) + 1
                for dx in range(i):
                    for dy in range(i):
                        if x + dx < m and y + dy < n:
                            matrix[x + dx][y + dy] = True  # 假设修改为1，可以根据需要更改
            if save_mid:
                matrix[size1//2][size2//2] = False
            return matrix
        return modify_matrix(matrix, size1, size2, np.random.randint(0, max_num+1), max_size, save_mid=save_mid)

    def __len__(self):
        return self.datasets.shape[0]

    def __getitem__(self, item):
        x = self.datasets[item]
        if self.config.data.dwt:
            wvt = pywt.wavedec2(x.cpu(), wavelet=self.wavelet, level=self.level)
            e = np.concatenate([np.array(wvt[0], dtype=np.float32), np.array(wvt[1][0], dtype=np.float32), np.array(wvt[1][1], dtype=np.float32), np.array(wvt[1][2], dtype=np.float32)], 0)
            e = torch.from_numpy(e)
        else:
            e = x
        return x, e, self.targets[item]
