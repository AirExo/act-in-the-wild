import torch
import torch.nn as nn


class AutoregressiveActionDecoder(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        autoregressive_bins
    ):
        super(AutoregressiveActionDecoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.autoregressive_bins = autoregressive_bins
        if output_dim % autoregressive_bins != 0:
            raise AttributeError('Output dimension should be divisible by autoregressive bins!')
        self.output_dim_per_bin = output_dim // autoregressive_bins
        output_heads = []
        for i in range(self.autoregressive_bins):
            output_heads.append(nn.Linear(input_dim + self.output_dim_per_bin * i, self.output_dim_per_bin))
        self.output_heads = nn.ModuleList(output_heads)

    def forward(self, x):
        res = []
        for i in range(self.autoregressive_bins):
            y = self.output_heads[i](torch.cat([x, *res], dim = -1))
            res.append(y)
        res = torch.cat(res, dim = -1)
        return res
    