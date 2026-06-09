import numpy as np
import torch
import pywt, time
import matplotlib.pyplot as plt

plot = False

def padding_image(data, size):
    size1 = size // 2
    c, h, w = data.shape
    data = torch.concat([torch.zeros((c, size1, w)), data, torch.zeros((c, size-size1, w))], dim=1)
    h += size
    data = torch.concat([torch.zeros((c, h, size1)), data, torch.zeros((c, h, size-size1))], dim=2)
    return data
def get_input(data, batch_size, img_size, log, dwt=False, dtype=np.float32, wavelet='db1', level=1):
    global plot
#     data.dtype = dtype
    n2, n3 = data.shape[1:]
    mod = n3 - (img_size) // 20
    for j in range(n3 - img_size):
        imgs, dwts = [], []
        for k in range(n2 - img_size):
            imgs.append(np.array(data[:, k:k+img_size, j:j+img_size].tolist(), dtype=dtype))
            if dwt:
                wvt = pywt.wavedec2(imgs[-1], wavelet=wavelet, level=level)
                dwts.append(
                    np.concatenate([np.array(wvt[0], dtype=dtype), np.array(wvt[1][0], dtype=dtype),
                                    np.array(wvt[1][1], dtype=dtype), np.array(wvt[1][2], dtype=dtype)], 0)
                )
            else:
                dwts.append(data)
            if len(imgs) == batch_size:
                imgs = np.stack(imgs, 0)
                dwts = np.stack(dwts, 0)
                yield False, torch.from_numpy(imgs), torch.from_numpy(dwts)
                imgs, dwts = [], []
        if len(imgs) > 0:
            imgs = np.stack(imgs, 0)
            dwts = np.stack(dwts, 0)
            yield True, torch.from_numpy(imgs), torch.from_numpy(dwts)
        else:
            yield True, None, None
        if j % 3 == 0:
            plot = True
            with open(log, 'a') as f:
                num = j / (n3 - img_size)
                formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                f.write(f'{formatted_time}\t{j} / {n3 - img_size} ({int(num * 10000) / 100}%)\n')
def classification(im, net, batch_size, img_size, log, save_path='', ddtype=np.float32, wavelet='db1', level=1, device=torch.device("cpu")):
    global plot
    #net = torch.load(model_path)
    image, pres, part1, part2 = [], [], [], []
    Print_shape = True
    data_path1 = save_path[:-4] + '_result'
    data_path2 = save_path[:-4] + '_pres'
    savei = 0
#     print(im.shape)
    for ii, (enter, ima, dwts) in enumerate(
            get_input(im, batch_size=batch_size, img_size=img_size, log = log,
                      dtype=ddtype, wavelet=wavelet, level=level)
    ):
        if ima is not None:
            ima, dwts = ima.to(device), ima.to(device)
            pre = net(ima)
            result = torch.argmax(
                pre
                , -1
            )
            if result.device.type != 'cpu':
                result = result.cpu()
                pre = pre.cpu()
            part1.append(result)
            part2.append(pre)
        if plot and len(image) > 0:
            try:
                plt.imshow(torch.stack(image, axis=-1))
                plt.savefig(save_path[:-4] + str(savei) + '.png')
                plt.close()
                torch.save(torch.stack(image, axis=-1), data_path1 + str(savei) + '.pt')
                torch.save(torch.stack(pres, axis=-1), data_path2 + str(savei) + '.pt')
                savei += 1
                del image
                del pres
                image, pres = [], []
            except:
                with open(log, 'a') as f:
                    f.write('中途绘制出现错误...\n')
            plot = False
        if enter:
            try:
                if len(part1) > 0 and len(part2) > 0:
                    part1 = torch.cat(part1, axis=-1)
                    part2 = torch.cat(part2, axis=-2)
                    image.append(part1)
                    pres.append(part2)
                else:
                    print('换行位置出现错误')
            except:
                print('开始分类...')
                with open(log, 'a') as f:
                    f.write('开始分类...\n')
            del part1
            del part2
            part1, part2 = [], []
        #                 start = time.time()
    try:
        if len(part1) > 0:
            part1 = torch.cat(part1, axis=-1)
            part2 = torch.cat(part2, axis=-1)
            image.append(part1)
            pres.append(part2)
        pres = torch.stack(pres, axis=-1)
        image = torch.stack(image, axis=-1)
    except:
        print('part部分出现错误...')
        with open(log, 'a') as f:
            f.write('part部分出现错误...\n')
        return part1, part2, image, pres
    if len(save_path) > 0:
        save_path = save_path[:-4] + str(savei) + '.png'
        data_path1 += str(savei) + '.pt'
        data_path2 += str(savei) + '.pt'
        try:
            with open(log, 'a') as f:
                f.write(f'图像绘制保存文件：{save_path}\n')
            plt.imshow(image)
            plt.savefig(save_path)
            plt.close()
            '''path = save_path.split('/')
            save_path = ''
            for i in range(len(path) - 1):
                save_path += path[i] + '/'
            save_path += path[-1].split('.') + '.pt'
'''
            with open(log, 'a') as f:
                f.write(f'数据保存文件：{data_path1}\n')
            torch.save(image, data_path1)
            with open(log, 'a') as f:
                f.write(f'数据保存文件：{data_path2}\n')
            torch.save(pres, data_path2)
        except:
            print('无法保存图像...')
            with open(log, 'a') as f:
                f.write('无法保存图像...\n')
            return None, None, image, pres
    return None, None, None, None

if __name__ == '__main__':
    im = padding_image(torch.rand((10, 29, 24)), 13)
    classification(im, None, 16, 13)
