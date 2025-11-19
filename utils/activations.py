import crypten.nn as cnn
import torch
import math


class silu(cnn.Module):
    def __init__(self):
        super().__init__()
        self.func = cnn.Sigmoid()

    def forward(self, x):
        return x * self.func(x)


class puma_gelu(cnn.Module):
    def __init__(self):
        super().__init__()

    def poly0(self, x):
        x2 = x.square()
        x3 = x * x2
        return (
            -0.011034134030615728 * x3
            - 0.11807612951181953 * x2
            - 0.42226581151983866 * x
            - 0.5054031199708174
        )

    def poly1(self, x):
        x2 = x.square()
        x3 = x * x2
        x4 = x2.square()
        x6 = x3.square()
        return (
            0.0018067462606141187 * x6
            - 0.037688200365904236 * x4
            + 0.3603292692789629 * x2
            + 0.5 * x
            + 0.008526321541038084
        )

    def forward(self, x):
        c0 = x < -4
        c1 = x < -1.95
        c2 = x.le(3)

        z0 = c0
        z1 = c1 - c0
        z2 = c2 - c1
        z3 = 1 - c2
        return 0 * z0 + z1 * self.poly0(x) + z2 * self.poly1(x) + z3 * x


class activation_newGeLU(cnn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def __init__(self):
        super().__init__()
        self.half = torch.tensor([0.5]).item()
        self.one = torch.tensor([1.0]).item()
        self.three = torch.tensor([3.0]).item()
        self.constant = torch.tensor([0.044715]).item()
        self.pi_const = torch.tensor([math.sqrt(2 / math.pi)]).item()
        self.pow = cnn.Pow()

    def forward(self, x):
        return (
            self.half
            * x
            * (
                self.one
                + (
                    self.pi_const * (x + self.constant * self.pow((x, self.three)))
                ).tanh()
            )
        )