import torch
import torch.nn as nn
from einops import rearrange

class VectorFeatureFutionModule(nn.Module):
    '''
    class token 特征压缩模块
        数量加倍
        特征减半
    '''
    def __init__(self, in_num, new_num, in_dim, mid_dims, out_dim, res_relu=True, p=0.5, split_class=0):
        '''
            Args:
            in_num: 输入数量
            new_num: 输出数量
            in_dim: 输入维度
            mid_dims:
            out_dim: 输出维度
            res_relu:
            p:
            split_class: 是否分离类别token
        '''
        super().__init__()
        self.split_class = split_class
        temp_dim = max(1, in_dim//4)
        self.old_embed = nn.Sequential(
            nn.Linear(in_dim, temp_dim),
            nn.Dropout(p=p),
            nn.ReLU(),
            nn.Linear(temp_dim, in_dim),
            nn.Dropout(p=p),
            nn.ReLU(),
        )
        modules = []
        if len(mid_dims) == 0:
            mid_dims = [in_dim // 2 if in_dim // 2 > 0 else 1, in_dim, in_dim * 2]
        for dim in mid_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_dim, dim),
                    nn.Dropout(p=p),
                    nn.ReLU(),
                    nn.Linear(dim, out_dim),
                    nn.Dropout(p=p),
                    nn.ReLU() if res_relu else nn.Identity(),
                ) if dim != out_dim else nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.Dropout(p=p),
                    nn.ReLU() if res_relu else nn.Identity(),
                )
            )
        self.my_modules = nn.ModuleList(modules)
        self.gen_new_token = nn.Sequential(
            nn.Linear(2*len(mid_dims)*in_num, new_num),
            nn.Dropout(p=p),
            nn.ReLU()
        )
    def forward(self, oldT, newT):
        x = torch.cat([self.old_embed(oldT), newT], dim=2 if self.split_class>0 else 1)
        out = []
        for module in self.my_modules:
            out.append(module(x))
        if self.split_class > 0:
            out = torch.cat(out, 2)
            b = out.shape[0]
            out = rearrange(out, 'b n c d -> (b n) d c')
            out = rearrange(self.gen_new_token(out), '(b n) c d -> b n d c', b=b)
        else:
            out = rearrange(torch.cat(out, 1), 'b c d -> b d c')
            out = rearrange(self.gen_new_token(out), 'b c d -> b d c')
        return out

class BaseExtractModule(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, conv_size):
        super().__init__()
        self.out_channel = out_channel
        if isinstance(mid_channel, int):
            mid_channel = [mid_channel]
        self.emd_modules = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channel, mid_chn, conv_size, padding=0 if conv_size == 1 else 1),
                    nn.BatchNorm2d(mid_chn),
                    nn.ReLU()
                ) for mid_chn in mid_channel
            ]
        )
        self.extract_module = nn.Sequential(
            nn.Conv2d(sum(mid_channel), 3*out_channel, conv_size, padding=0 if conv_size==1 else 1),
            nn.BatchNorm2d(3*out_channel)
        )
    def forward(self, x):
        mid_feature = torch.cat([module(x) for module in self.emd_modules], dim=1)
        feature = self.extract_module(mid_feature)
        return nn.Sigmoid()(feature[:, :self.out_channel]) * feature[:, self.out_channel:2*self.out_channel] + feature[:, 2*self.out_channel:]

class BaseOldNewTokenFution(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, embed_size=1):
        super().__init__()
        self.mid_channel = mid_channel
        self.out_channel = out_channel
        self.embed = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, 1),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(),
            nn.Conv2d(mid_channel, mid_channel, embed_size, padding=0 if embed_size == 1 else 1),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU()
        )
        self.extract = BaseExtractModule(mid_channel, mid_channel, out_channel, embed_size)

    def forward(self, x):
        temp = self.embed(x)
        return self.extract(temp)

class Feature2Token(nn.Module):
    def __init__(self, in_channel, embed_channel, out_channel, in_dim, mid_dim, out_dim, p=0.2, split_class=0):
        '''
        Args:
            in_channel: 输入图像通道数
            embed_channel:
            out_channel: 输出token数量
            in_dim: 图像空间向量长度
            mid_dim:
            out_dim:输出token维度
            split_class:输出分类数量
            p:
        '''
        super().__init__()
        self.split_class = split_class
        if split_class > 0:
            out_channel = split_class * out_channel
        self.embed = nn.Sequential(
            nn.Conv2d(in_channel, embed_channel, 3, 1, 1),
            nn.BatchNorm2d(embed_channel),
            nn.ReLU(),
            nn.Conv2d(embed_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.module = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.Dropout(p=p),
            nn.ReLU(),
            nn.Linear(mid_dim, out_dim),
            nn.Dropout(p=p),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.embed(x)
        x = rearrange(x, 'b c h w -> b c (h w)')
        x = self.module(x)
        if self.split_class > 0:
            x = rearrange(x, 'b (n c) d -> b n c d', n=self.split_class)
        return x

class CITF(nn.Module):
    '''
    in:
          last_img_token
           new_img_token
        last_class_token
         new_class_token
    output:
               img_token
             class_token
    '''
    def __init__(self, in_dim, token_num, in_class_num, new_class_num, out_dim=None, mid_dims=[], ps=[0.25, 0.1],
                 split_class=0, use_spec_token=False, use_spat_token=False):
        '''
        Args:
            in_dim: old spec feature channel + now spec feature channel == one feature channel
            token_num: == height * weigth
            in_class_num: 输入数量
            new_class_num: 输出数量
            out_dim: new feature channel
            mid_dims: -
            ps:
            split_class:
        '''
        super().__init__()
        self.use_spec_token = use_spec_token
        self.use_spat_token = use_spat_token
        if split_class:
            # 当分类别学习class token时保证可以按照输出数量可分
            self.multiple = round(new_class_num / in_class_num)
            new_class_num = in_class_num * self.multiple
        new_class_num = new_class_num + in_class_num
        self.split_class = split_class
        if out_dim is None:
            out_dim = in_dim // 2
        out_dim = max(1, out_dim)
        # 生成全新的特征
        spec_channel_num, spat_channel_num = self._split_channel(out_dim)
        if use_spec_token:
            # 旧光谱+新光谱维度， 中间随意维度， 输出维度， 光谱卷积核大小
            self.spec_class_img_token_fution_module = BaseOldNewTokenFution(in_dim*2, max(4, in_dim // 2),
                                                                            spec_channel_num, embed_size=1)
        if use_spat_token:
            # 旧空间+新空间维度， 中间随意维度， 输出维度， 光谱卷积核大小
            self.spat_class_img_token_fution_module = BaseOldNewTokenFution(in_dim if use_spec_token else in_dim * 2, max(4, in_dim // 4),
                                                                            spat_channel_num, embed_size=3)
        # 新旧class token融合
        # 输入数量 新token数量 输入维度 —— 输出维度
        self.class_token_fution_module = VectorFeatureFutionModule(in_class_num, new_class_num//2, in_dim, mid_dims, out_dim, p=ps[0], split_class=split_class)
        # 根据新旧特征提取新的class token用于特征补充
        temp_dim = out_dim + in_dim
        # 输入图像通道数 - 输出token数量 图像空间向量长度 - 输出token维度
        self.new_class_token_gen_module = Feature2Token(temp_dim, temp_dim//4, new_class_num,
                token_num, token_num//4, out_dim, p=0.2, split_class=split_class)
    def forward(self, old_img_feature, old_class_token, now_img_feature, now_class_token):
        c1, _ = self._split_channel(old_img_feature.shape[1])
        c2, _ = self._split_channel(now_img_feature.shape[1])
        h, w = old_img_feature.shape[-2:]
        new_img_feature = []
        if self.use_spec_token:
            spec_feature = torch.cat([old_img_feature[:, :c1], now_img_feature[:, :c2],
                                      old_img_feature[:, c1:, h//2, w//2].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, h, w),
                                      now_img_feature[:, c1:, h//2, w//2].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, h, w)], dim=1)
            spec_feature = self.spec_class_img_token_fution_module(spec_feature)
            new_img_feature.append(spec_feature)
        if self.use_spat_token:
            spat_feature = torch.cat([old_img_feature[:, c1:], now_img_feature[:, c2:]], dim=1)
            spat_feature = self.spat_class_img_token_fution_module(spat_feature)
            new_img_feature.append(spat_feature)
        new_img_feature = torch.concat(new_img_feature, dim=1)
        new_class_token = self.class_token_fution_module(old_class_token, now_class_token)
        new_class_token_add = self.new_class_token_gen_module(torch.cat([old_img_feature, new_img_feature], dim=1))
        if self.split_class > 0:
            n = new_class_token.shape[2]
            new_class_token = new_class_token + new_class_token_add[:, :, :n]
            new_class_token = self._alternative_splicing(new_class_token, new_class_token_add[:, :, n:])
        else:
            n = new_class_token.shape[1]
            new_class_token = new_class_token + new_class_token_add[:, :n]
            new_class_token = torch.concat(
                [new_class_token, new_class_token_add[:, n:]], -2
            )
        # if self.split_class > 0:
        #     b = new_class_token.shape[0]
        #     new_class_token = rearrange(new_class_token, 'b n d k -> (b n) d k')
        # new_class_token = self.class_token_fution(new_class_token)
        # if self.split_class > 0:
        #     new_class_token = rearrange(new_class_token, '(b n) d k -> b (n k) d', b=b)
        return new_img_feature, new_class_token
    def _alternative_splicing(self, x1, x2):
        shape1, shape2 = x1.shape, x2.shape
        assert (shape1[0] == shape2[0] and shape1[2] == shape2[2] and shape1[3] == shape2[3])
        result = torch.empty((shape2[0], shape2[1], shape1[2]+shape2[2], shape2[3]))
        result[:, :, ::(self.multiple+1)] = x1
        for i in range(self.multiple):
            result[:, :, (i+1)::(self.multiple+1)] = x2[:, :, i::self.multiple]
        return result

    def _split_channel(self, in_channel):
        assert self.use_spec_token or self.use_spat_token
        if self.use_spec_token:
            if self.use_spat_token:
                return in_channel // 2, in_channel - in_channel // 2
            else:
                return in_channel, 0
        return 0, in_channel



'''
from thop import profile
if __name__ == '__main__':
    split_class = 1
    model = CITF(100, 169, 10, 20, split_class=split_class)
    im1 = torch.randn((64, 100, 13, 13))
    im2 = torch.randn((64, 100, 13, 13))
    c1 = torch.randn((64, split_class, 10, 100))
    c2 = torch.randn((64, split_class, 10, 100))
    y1, y2 = model(im1, c1, im2, c2)
    # print(y1.shape, y2.shape)
    # model = CITF(128, 56, 9)
    # spec_class_token = torch.randn((64, 9, 128))
    # old_feature = torch.randn((64, 56, 128))
    # new_feature = torch.randn((64, 56, 128))
    flops, params = profile(model, (im1, c1, im2, c2))
    print('flops: ', flops, 'params: ', params)
    # y = model(spec_class_token, spec_feature, spat_class_token, spat_feature)
    print(y1.shape, y2.shape)
'''
