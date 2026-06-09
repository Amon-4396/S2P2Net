import torch
import numpy as np
import torch.nn as nn
from einops import rearrange
from transformer import Transformer, CrossAttention
from class_img_token_fution_module import CITF

#重点1 如何有效利用中心像素
#重点2 如何实现有效融合class token和normal token

class CenterSpecRestruct(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False, mid_dim=0):
        super().__init__()
        if mid_dim == 0:
            mid_dim = self.get_mid_dim(dim, num_heads)
        self.in_embedding1 = self.make_embedding(dim, mid_dim, mid_dim, layer=1)
        self.in_embedding2 = self.make_embedding(dim, mid_dim, mid_dim, layer=1)
        self.attn = CrossAttention(mid_dim, num_heads, bias)
        self.out_embedding = self.make_embedding(mid_dim, dim, dim, layer=1)
        self.to_class_token = nn.Identity()


    def forward(self, x1, x2):
        # x1 -> q
        # x2 -> k v
        temp1 = rearrange(x1, 'b c -> b 1 c')
        temp2 = rearrange(x2, 'b c h w -> b (h w) c')
        for module in self.in_embedding1:
            temp1 = module(temp1)
        for module in self.in_embedding2:
            temp2 = module(temp2)

        out = self.attn(temp1, temp2)  # b, c, dim
        for module in self.out_embedding:
            out = module(out)
        return rearrange(out, 'b n d -> b (n d)')

    def make_embedding(self, in_dim, mid_dim, end_dim, layer=1):
        ret = []
        dim = in_dim
        while 1:
            if layer == 1:
                ret.append(nn.Linear(dim, end_dim))
                ret.append(nn.Dropout())
                ret.append(nn.ReLU())
                break
            ret.append(nn.Linear(dim, mid_dim))
            ret.append(nn.Dropout())
            ret.append(nn.ReLU())
        return nn.ModuleList(ret)

    def get_mid_dim(self, dim, num_heads):
        return (dim // num_heads + 1) * num_heads
# 737937
class EmbeddingModule(nn.Module):
    def __init__(self, in_channel, out_channel, use_spec_token=False, use_spat_token=False):
        super().__init__()
        assert use_spec_token or use_spat_token
        self.use_spec_token = use_spec_token
        self.use_spat_token = use_spat_token
        # 初始嵌入，减少参数
        # in_channel = self._make_base_embed(in_channel, max(8, self._make_base_embed(in_channel, out_channel) // 8))
        in_channel = self._make_base_embed(in_channel, self._make_base_embed(in_channel, out_channel))

        if use_spat_token:
            if use_spec_token: # 光谱/空间均分
                spec_channel = out_channel // 2
                spat_channel = out_channel - spec_channel
            else: # 仅包含空间特征
                spat_channel = out_channel
        else: # 仅包含光谱特征
            spec_channel = out_channel
        if use_spat_token:
            self._make_spat_embed(in_channel,
                [max(2, in_channel // 16), max(2, in_channel // 8), max(2, in_channel // 4), max(2, in_channel // 2)],
                                  spat_channel)
        if use_spec_token:
            self._make_sec_embed(in_channel,
            [max(2, in_channel // 16), max(2, in_channel // 8), max(2, in_channel // 4), max(2, in_channel // 2)],
                                 spec_channel)
    def forward(self, x, y):
        x, y = self.base_spat_embed(x), self.base_spec_embed(y)
        out = []
        h, w = x.shape[2:]
        if self.use_spec_token:
            y = self.embed_spec(y, x) + y
            y = y.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, h, w)
            spec_feature = x - y
            spec_feature = self.extract_spec1(torch.concat([module(spec_feature) for module in self.se_spec], 1)) + y
            spec_map = self.gen_spat_map(spec_feature)
            spec_feature = spec_feature * spec_map[:, 0].unsqueeze(1) + y * spec_map[:, 1].unsqueeze(1)
            spec_feature = self.extract_spec2(spec_feature)
            out.append(spec_feature)
        if self.use_spat_token:
            spat_feature = self.extract_spat1(torch.concat([module(x) for module in self.se_spat], 1)) + x
            spat_feature = self.extract_spat2(spat_feature)
            out.append(spat_feature)
        return torch.cat(out, dim=1)

    def _make_base_embed(self, in_channel, out_channel):
        self.base_spat_embed = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.base_spec_embed = nn.Sequential(
            nn.Linear(in_channel, out_channel),
            nn.Dropout(p=0.1),
            nn.ReLU()
        )
        return out_channel
    def _make_spat_embed(self, in_channel, mid_channels, spat_channel):
            self.se_spat = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(in_channel, mid_channel, 3, 1, 1),
                    nn.BatchNorm2d(mid_channel),
                    nn.ReLU(),
                    nn.Conv2d(mid_channel, mid_channel, 3, 1, 1),
                    nn.BatchNorm2d(mid_channel),
                    nn.ReLU(),
                    nn.Conv2d(mid_channel, in_channel//len(mid_channels), 1),
                    nn.BatchNorm2d(in_channel//len(mid_channels)),
                    nn.ReLU(),
                ) for mid_channel in mid_channels
            ])
            self.extract_spat1 = nn.Sequential(
                nn.Conv2d(in_channel, in_channel, 1),
                nn.BatchNorm2d(in_channel),
                nn.ReLU()
            )
            self.extract_spat2 = nn.Sequential(
                nn.Conv2d(in_channel, spat_channel, 1),
                nn.BatchNorm2d(spat_channel),
                nn.ReLU()
            )
    def _make_sec_embed(self, in_channel, mid_channels, spec_channel):
            self.embed_spec = CenterSpecRestruct(in_channel)
            self.se_spec = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(in_channel, mid_channel, 1),
                    nn.BatchNorm2d(mid_channel),
                    nn.ReLU(),
                    nn.Conv2d(mid_channel, in_channel // len(mid_channels), 1),
                    nn.BatchNorm2d(in_channel // len(mid_channels)),
                    nn.ReLU(),
                ) for mid_channel in mid_channels
            ])
            self.extract_spec1 = nn.Sequential(
                nn.Conv2d(in_channel, in_channel, 1),
                nn.BatchNorm2d(in_channel),
                nn.ReLU()
            )
            self.gen_spat_map = nn.Sequential(
                nn.Conv2d(in_channel, 2, 1),
                nn.BatchNorm2d(2),
                nn.ReLU()
            )
            self.extract_spec2 = nn.Sequential(
                nn.Conv2d(in_channel, spec_channel, 1),
                nn.BatchNorm2d(spec_channel),
                nn.ReLU()
            )

class BaseMLP(nn.Module):
    def __init__(self, in_ch, out_ch, device, soft=True):
        super(BaseMLP, self).__init__()
        self.model = nn.ModuleList(
            self.make_mlp(in_ch, out_ch, soft)
        ).to(device)
    def forward(self, x):
        for module in self.model:
            x = module(x)
        return x
    def make_mlp(self, in_ch, out_ch, soft):
        ret = []
        n = 2
        while n < in_ch:
            n *= 2
        if n - in_ch > in_ch - n // 2:
            n = n // 2
        while in_ch > 32:
            ret.append(nn.Linear(in_ch, n // 4))
            ret.append(nn.Dropout(0.5))
            ret.append(nn.ReLU())
            in_ch = n // 4
            n = n // 4
        ret.append(nn.Linear(in_ch, out_ch))
        if soft:
            ret.append(nn.Softmax(dim=1))
        else:
            ret.append(nn.ReLU())
        # print('mlp:\n', ret)
        return ret

class ScoreModule(nn.Module):
    def __init__(self, in_dim, class_size, split_n, device):
        super(ScoreModule, self).__init__()
        self.vote = nn.ModuleList([
            nn.Sequential(
                nn.Flatten(),
                BaseMLP(in_dim, 1, device, soft=False)
            ) for _ in range(split_n)
        ])
        if class_size != split_n:
            self.fution = nn.Sequential(
                nn.Linear(split_n, class_size),
                nn.Softmax(dim=-1)
            ).to(device)
        else:
            self.fution = nn.Sigmoid()
    def forward(self, class_token):
        pre = []
        for i in range(class_token.shape[1]):
            temp = class_token[:, i]
            pre.append(self.vote[i](temp))
        pre = torch.concat(pre, -1)
        return self.fution(pre)

class O3RC(nn.Module):
    def __init__(self, in_channel, class_num, size1, size2, device, split_class=False, share_mlp=False, share_vote=False):
        super().__init__()
        self.share_mlp = share_mlp
        self.share_vote = share_vote
        self.one_class_token = not split_class
        if share_mlp or self.one_class_token:
            self.mlp = self._make_conv(size1, size2, in_channel)
        else:
            self.mlp = []
            for i in range(class_num):
                self.mlp.append( self._make_conv(size1, size2, in_channel) )
            self.mlp = nn.ModuleList(self.mlp)
        # in_ch_size = in_channel//2//2 * ((self.size1 - 3) // 2 + 1) * ((self.size2 - 3) // 2 + 1)
        if self.one_class_token:
            self.voters = nn.Linear(self.in_ch_size, class_num)
            self.activation_function = nn.Softmax(dim=-1)
        elif share_vote:
            self.voters = nn.Linear(self.in_ch_size, 1)
            self.activation_function = nn.Sigmoid()
        else:
            self.params = nn.Parameter(torch.ones(class_num, self.in_ch_size))
            nn.init.kaiming_uniform_(self.params)
            self.bias = nn.Parameter(torch.zeros(class_num))
            self.activation_function = nn.Sigmoid()

    def forward(self, class_feature):
        out1, out2 = [], []
        if self.one_class_token:
            out1 = self.mlp(class_feature)
        elif self.share_mlp:
            for token in class_feature:
                out1.append(self.mlp(token))
            out1 = torch.stack(out1, dim=1)
        else:
            for i, module in enumerate(self.mlp):
                out1.append(module(class_feature[i]))
            out1 = torch.stack(out1, dim=1)
        if self.one_class_token:
            out2 = self.voters(out1)
        elif self.share_vote:
            for feature in out1:
                out2.append(self.voters(feature))
            out2 = torch.cat(out2, dim=-1)
        else:
            feature = self.params.unsqueeze(0) * out1
            out2 = torch.sum(feature, -1) + self.bias.unsqueeze(0)
        out = self.activation_function(out2)
        return out

    def _make_conv1(self, size1, size2, in_channel):
        ret = []
        i = 0
        use_3 = True
        mid_channel = max(2, in_channel // 8)
        ret.append(nn.Conv2d(in_channel, mid_channel, 1, 1))
        ret.append(nn.BatchNorm2d(mid_channel))
        ret.append(nn.ReLU())
        ret.append(nn.Conv2d(in_channel, mid_channel, 3, 1))
        ret.append(nn.BatchNorm2d(mid_channel))
        ret.append(nn.ReLU())
        size1, size2 = size1 - 2, size2 - 2
        self.in_ch_size = size1 * size2 * in_channel
        ret.append(nn.Flatten())
        return nn.Sequential(*ret)
    def _make_conv(self, size1, size2, in_channel):
        ret = []
        i = 0
        use_3 = True
        while in_channel > 8:
            if use_3 and (size1 < 3 or size2 < 3):
                use_3 = False
            mid_channel = in_channel // 8
            if i % 2 == 0 or not use_3:
                ret.append(nn.Conv2d(in_channel, mid_channel, 1, 1))
                ret.append(nn.BatchNorm2d(mid_channel))
                ret.append(nn.ReLU())
            else:
                ret.append(nn.Conv2d(in_channel, mid_channel, 3, 1))
                ret.append(nn.BatchNorm2d(mid_channel))
                ret.append(nn.ReLU())
                size1, size2 = size1 - 2, size2 - 2
            i += 1
            in_channel = mid_channel
        self.in_ch_size = size1 * size2 * in_channel
        ret.append(nn.Flatten())
        return nn.Sequential(*ret)

class FeatureExtractionModule(nn.Module):
    def __init__(self, config, now_dataset, in_size, split_class=0, device=torch.device('cpu')):
        super().__init__()
        self.config = config
        split_class = split_class if split_class > 1 else 0
        self.split_class = split_class
        self.device = device
        self.use_spec_token = config.ablation.v1.spec_token
        self.use_spat_token = config.ablation.v1.spat_token
        self.class_image_fution = config.ablation.v2.class_image_fution # 是否使用类别图像融合模块
        assert not(not self.class_image_fution and config.ablation.v2.change_token_dim)
        layers = config.model.layers

        if now_dataset.pca:
            print('使用PCA降维')
        self.out_channel = (layers[-1][3]+layers[-1][2]) * config.model.input_dim_multiple_num * \
                           layers[-1][4] * (len(layers) if config.model.class_use_old_feature else 1) // layers[-1][-1]
        '''with open('./error/log.txt', 'a') as f:
            f.write('layers[-1][3]+layers[-1][2]: ')
            f.write(str(layers[-1][3]+layers[-1][2]))
            f.write('\n')
            f.write('layers[-1][-1]: ')
            f.write(str(layers[-1][-1]))
            f.write('\n')
            f.write('config.model.input_dim_multiple_num: ')
            f.write(str(config.model.input_dim_multiple_num))
            f.write('\n')
            f.write('layers[-1][4]: ')
            f.write(str(layers[-1][4]))
            f.write('\n')
            f.write('beishu : ')
            f.write(str(len(layers) if config.model.class_use_old_feature else 1))
            f.write('\n')'''
        # if not self.class_image_fution:
        if config.ablation.v2.change_token_dim:
            self.attn_feature_extraction = nn.ModuleList([
                # 1.输入token维度 2.多头注意力数量
                Transformer(layer_config[0] // layers[-1][-1] * config.model.input_dim_multiple_num,
                            layer_config[1]) for layer_config in layers
            ])
        else:
            self.attn_feature_extraction = nn.ModuleList([
                # 1.仅使用初始特征 2.多头注意力数量
                Transformer(layers[0][0] // layers[-1][-1] * config.model.input_dim_multiple_num,
                            layer_config[1]) for layer_config in layers
            ])

        if self.class_image_fution:
            if config.ablation.v2.change_token_dim:
                self.feature_fuction = nn.ModuleList([
                    # 1.输入token维度，空间像元个数，- , 3.输入class token数量，4.输出增加class token数量 5.输出class token维度
                    CITF(layers[i][0] // layers[-1][-1] * config.model.input_dim_multiple_num,
                    self.config.model.img_patch.patch_size[0] * self.config.model.img_patch.patch_size[1] \
                        if self.config.model.img_patch.use else in_size[0] * in_size[1], layers[i][2], layers[i][3],
                         out_dim=layers[i][4] // layers[-1][-1] * config.model.input_dim_multiple_num,
                         split_class=split_class, use_spec_token=self.use_spec_token, use_spat_token=self.use_spat_token)
                    for i in range(len(layers)-1)
                ])
            else:
                self.feature_fuction = nn.ModuleList([
                    # 1.输入token维度，空间像元个数，- , 3.输入class token数量，4.输出class token数量 5.输出class token维度
                    CITF(layers[0][0] // layers[-1][-1] * config.model.input_dim_multiple_num,
                    self.config.model.img_patch.patch_size[0] * self.config.model.img_patch.patch_size[1] \
                        if self.config.model.img_patch.use else in_size[0] * in_size[1],
                         layers[i][2], layers[i][3], out_dim=layers[0][0] // layers[-1][-1] * config.model.input_dim_multiple_num,
                         split_class=split_class, use_spec_token=self.use_spec_token, use_spat_token=self.use_spat_token)
                    for i in range(len(layers)-1)
                ])
                self.out_channel = sum([layer_config[2] for layer_config in layers]) * layers[0][0] * config.model.input_dim_multiple_num // layers[-1][-1]
    def forward(self, old_img_feature, old_class_token):
        h, w = old_img_feature.shape[-2:]

        class_features = []
        for i in range(len(self.attn_feature_extraction)):
            old_class_token = old_class_token.to(self.device)
            old_img_feature = rearrange(old_img_feature, 'b n h w -> b (h w) n')
            if self.split_class:
                k = old_class_token.shape[1]
                old_class_token = rearrange(old_class_token, 'b k n d -> b (k n) d')
            new_feature_token = self.attn_feature_extraction[i](old_img_feature, old_class_token)
            new_class_token, new_img_feature = new_feature_token[:, :old_class_token.shape[1]], new_feature_token[:, old_class_token.shape[1]:]
            if self.split_class:
                new_class_token = rearrange(new_class_token, 'b (k n) d -> b k n d', k=k)
                old_class_token = rearrange(old_class_token, 'b (k n) d -> b k n d', k=k)
            old_img_feature = rearrange(old_img_feature, 'b (h w) n -> b n h w', h=h)
            new_img_feature = rearrange(new_img_feature, 'b (h w) c -> b c h w', h=h)

            # 保存类别特征
            if self.config.model.class_use_old_feature:
                class_features.append(new_class_token.reshape(new_class_token.shape[0], new_class_token.shape[1], -1) if self.split_class else
                                      new_class_token.reshape(new_class_token.shape[0], -1))
            else:
                class_features = [new_class_token.reshape(new_class_token.shape[0], new_class_token.shape[1], -1) if self.split_class else
                                      new_class_token.reshape(new_class_token.shape[0], -1)]
            # token增加融合
            if self.class_image_fution and i < len(self.feature_fuction):
                old_img_feature, old_class_token = self.feature_fuction[i](old_img_feature, old_class_token,
                                                                           new_img_feature, new_class_token)
            else:
                old_img_feature, old_class_token = new_img_feature, new_class_token
        class_features = torch.concat(class_features, -1).to(self.device)
        return class_features

class Model(nn.Module):
    def __init__(self, config, now_dataset, in_ch, in_size, class_size, split_class=0,
                 device=torch.device('cpu'), share_mlp=False, share_vote=False):
        super().__init__()
        split_class = split_class if split_class > 1 else 0
        self.split_class = split_class
        layers = config.model.layers
        self.many_image_patch = config.model.img_patch.use

        if self.many_image_patch:
            self.in_size = in_size
            self.patch_size, self.stride = config.model.img_patch.patch_size, config.model.img_patch.stride
            self.left1  = (self.in_size[0] // 2 - self.patch_size[0] // 2) // self.stride[0]
            self.right1 = ((self.in_size[0] - self.in_size[0] // 2 + self.patch_size[0] // 2) - self.patch_size[0]) // self.stride[0] + 1
            self.left2  = (self.in_size[1] // 2 - self.patch_size[1] // 2) // self.stride[1]
            self.right2 = ((self.in_size[1] - self.in_size[1] // 2 + self.patch_size[1] // 2) - self.patch_size[1]) // self.stride[1] + 1

        if split_class > 1:
            self.class_token = nn.Parameter(torch.zeros((1, split_class, layers[0][2],
                                                layers[0][0] // layers[-1][-1] * config.model.input_dim_multiple_num)))
        else:
            self.class_token = nn.Parameter(torch.zeros((1, layers[0][2],
                                                layers[0][0] // layers[-1][-1] * config.model.input_dim_multiple_num)))
        if now_dataset.pca:
            print('使用PCA降维')
        # 初始嵌入模块
        self.embed = EmbeddingModule(in_ch, layers[0][0] // layers[-1][-1] * config.model.input_dim_multiple_num,
                                     use_spec_token=config.ablation.v1.spec_token, use_spat_token=config.ablation.v1.spat_token)
        self.fe = FeatureExtractionModule(config, now_dataset, in_size, split_class, device)
        if self.many_image_patch:
            size1 = self.right1 + self.left1
            size2 = self.right2 + self.left2
            self.mlp = O3RC(self.fe.out_channel, class_size, size1, size2, device, split_class=split_class, share_mlp=share_mlp,
                            share_vote=share_vote).to(device)
        elif self.split_class > 0:
            self.mlp = ScoreModule(self.fe.out_channel,
                                   class_size, self.split_class, device)
        else:
            self.mlp = BaseMLP(self.fe.out_channel,
                               class_size, device, soft=True)

    def forward(self, x):
        h, w = x.shape[-2:]
        old_img_feature = self.embed(x, x[..., h//2, w//2])
        if self.many_image_patch:
            old_img_feature = self._class_conv_process(old_img_feature, self.in_size, self.patch_size, self.stride)
            b, n, c1, h1, w1 = old_img_feature.shape
            old_img_feature = old_img_feature.reshape(b * n, c1, h1, w1)
        if self.split_class:
            old_class_token = self.class_token.expand(old_img_feature.shape[0], -1, -1, -1)
        else:
            old_class_token = self.class_token.expand(old_img_feature.shape[0], -1, -1)
        class_features = self.fe(old_img_feature, old_class_token)
        if self.many_image_patch:
            if self.split_class:
                class_features = class_features.reshape(b, self.right1 + self.left1, self.left2 + self.right2, class_features.shape[1], -1).permute(3, 0, 4, 1, 2)
            else:
                class_features = class_features.reshape(b, self.right1 + self.left1, self.left2 + self.right2, -1).permute(0, 3, 1, 2)
        pre = self.mlp(class_features)
        return pre

    def _class_conv_process(self, img, in_size, patch_size, stride):
        imgs = []
        size1 = in_size[0] // 2 - patch_size[0] // 2
        size2 = in_size[1] // 2 - patch_size[1] // 2
        for i in range(-self.left1, self.right1):
            for j in range(-self.left2, self.right2):
                # if i == 0 and j == 0:
                #     print('ok')
                m = size1 + i * stride[0]
                n = size2 + j * stride[1]
                xf = img[:, :, m:m+patch_size[0], n:n+patch_size[1]]
                assert(xf.shape[-2] == patch_size[0] and xf.shape[-1] == patch_size[1])
                imgs.append(xf)
        imgs = torch.stack(imgs, dim=1)
        return imgs

# from thop import profile
# import argparse, yaml, os, datetime
# import torch.utils.tensorboard as tb
# def dict2namespace(config):
#     namespace = argparse.Namespace()
#     for key, value in config.items():
#         if isinstance(value, dict):
#             new_value = dict2namespace(value)
#         else:
#             new_value = value
#         setattr(namespace, key, new_value)
#     return namespace
# def set_init():
#     parser = argparse.ArgumentParser(description=globals()["__doc__"])
#     parser.add_argument('-config', help='Set configs file', default="cfg.yml", type=str)
#     parser.add_argument("--resume_training", action="store_true", help="Whether to resume training", default=False)
#     parser.add_argument("--seed", help="Whether to resume training", default=10, type=int)
#     args = parser.parse_args()
#
#     with open(args.config, "r", encoding='utf-8') as f:
#         config = yaml.safe_load(f)
#     config = dict2namespace(config)
#
#     try:
#         os.mkdir('tensorboard')
#         os.mkdir('checkpoint')
#     except:
#         pass
#     tb_path = f"tensorboard/tb_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     checkpoint_dir = f"checkpoint/ckpt_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#
#     if not os.path.isdir(checkpoint_dir):
#         os.makedirs(checkpoint_dir)
#     args.checkpoint_dir = checkpoint_dir
#     tb_logger = tb.SummaryWriter(log_dir=tb_path)
#
#     #  ---  Set random seed   ---  #
#     torch.manual_seed(args.seed)
#     np.random.seed(args.seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(args.seed)
#
#     return args, config, tb_logger
#
# if __name__ == '__main__':
#     args, config, tb_logger = set_init()
#     datasetname = 'indian pines'
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     now_dataset = config.dataset.ip if datasetname == 'indian pines' \
#         else (config.dataset.pu if datasetname == 'pavia university' else config.dataset.whu)
#     model = Model(
#         config,
#         now_dataset,
#         now_dataset.pca_channel if now_dataset.pca else now_dataset.in_channel,
#         config.data.input_size,
#         now_dataset.class_num,
#         split_class=now_dataset.class_num if config.model.split_class else 0,
#         device=device
#     ).to(device)
#     # model.load_state_dict(torch.load('C:/Users/86188/Downloads/best_model_pavia university_0.0001.pth', map_location=device)[0])
#     # for i in range(model.params.shape[0]):
#     #     active = (model.params[i] > 0).sum()
#     #     nagetive = (model.params[i] < 0).sum()
#     #     all = model.params[i].shape[0]
#     #     print(f'{i}:  active-{active}({int(10000*active/all)/100}%)  nagetive-{nagetive}({int(10000*nagetive/all)/100}%) all-{all}')
#     inputs1 = torch.randn(64, now_dataset.pca_channel if now_dataset.pca else now_dataset.in_channel, 13, 13).to(device)
#
#     # e = torch.from_numpy(np.array(pywt.wavedec2(inputs1.cpu(), wavelet='db1', level=1)[0], dtype=np.float32)).cuda()
#     flops, params = profile(model, (inputs1, ))
#     print('flops: ', flops, 'params: ', params)
#
