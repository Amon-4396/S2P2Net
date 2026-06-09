import torch
import torch.nn as nn

class DisLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        b = input.shape[0]
        target = torch.arange(input.shape[0]) * input.shape[1] + target
        input = input.reshape(-1)
        x = input[target]
        input[target] = 0
        input = input.reshape(b, -1)
        y = torch.max(input, dim=1)[0]
        # print(x, y)
        # print(x.shape, y.shape)
        return torch.mean(torch.tan(1 - torch.abs(x - y)))

class LargeMarginLoss(nn.Module):
    def __init__(self, m=1):
        super().__init__()
        self.m = m
    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], -1)
        i1 = x.unsqueeze(1).repeat(1, x.shape[0], 1, 1)
        i2 = x.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        dis = torch.mean(torch.mean(torch.pow(i1 - i2, 2), -1), -1)
        same = torch.eye(x.shape[0], x.shape[0])
        return torch.max(torch.tensor(0), self.m - dis[same == 0].mean())
if __name__ == "__main__":
    x = torch.rand((64, 12))
    y = torch.tensor([1,2,3,4,5,6,7,8]).repeat(8)
    ls = DisLoss()
    ls(x, y)