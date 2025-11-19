import warnings
import crypten
from crypten.nn import Module
import crypten.common.functions.approximations as approx
from crypten.config import cfg
import crypten.mpc
import torch
from crypten.cuda.cuda_tensor import implements, CUDALongTensor
import crypten.common.rng as rng

# Effectively comments our lines 526 to 531 in crypten/nn/module.py
def __getattribute__(self, name):
    if name != "forward":
        return object.__getattribute__(self, name)

    def forward_function(*args, **kwargs):
        """
        Silently encrypted Torch inputs tensors (deprecated).
        """
        if self.encrypted and not self.SUPPORTS_PLAINTEXT_INPUTS:
            if any(torch.is_tensor(arg) for arg in args):
                warnings.warn(
                    "Earlier versions of CrypTen silently encrypted Torch tensors. "
                    "That behavior is now deprecated because it is dangerous. "
                    "Please make sure you feed your model CrypTensors when needed.",
                    DeprecationWarning,
                )
        return object.__getattribute__(self, name)(*tuple(args), **kwargs)

    return forward_function


def enable_public_weights():
    Module.__getattribute__ = __getattribute__


def piecewise_approx(tensor):
    default_approx = (
        approx.exp(tensor.div(2).add(0.2).neg())
        .mul(2.2)
        .add(0.2)
        .add(tensor.div(1024).neg())
    )
    result = default_approx
    result = crypten.where(tensor < 1e-2, 10.0, result)
    result = crypten.where(tensor < 1e-4, 100.0, result)
    result = crypten.where(tensor < 1e-6, 1000.0, result)
    result = crypten.where((tensor < 200).get_plain_text(), result, 0.03)
    result = crypten.where((tensor < 4000).get_plain_text(), result, 0.01)
    return result


def inv_sqrt(self):
    r"""
    Computes the inverse square root of the input using the Newton-Raphson method.

    Configuration params:
        sqrt_nr_iters (int):  determines the number of Newton-Raphson iterations to run.
        sqrt_nr_initial (tensor): sets the initial value for the Newton-Raphson iterations.
                    By default, this will be set to allow the method to converge over a
                    fairly large domain.

    .. _Newton-Raphson:
        https://en.wikipedia.org/wiki/Fast_inverse_square_root#Newton's_method
    """
    initial = cfg.functions.sqrt_nr_initial
    iters = cfg.functions.sqrt_nr_iters

    # Initialize using decent approximation
    if initial is None:
        y = piecewise_approx(self)
    else:
        y = initial

    # Newton Raphson iterations for inverse square root
    for _ in range(iters):
        y = y.mul_(3 - self * y.square()).div_(2)
    return y


def replace_approx():
    approx.inv_sqrt = inv_sqrt
    setattr(crypten.CrypTensor, "inv_sqrt", inv_sqrt)
    setattr(crypten.mpc.MPCTensor, "inv_sqrt", inv_sqrt)

@staticmethod
@implements(torch.matmul)
def matmul(x, y, *args, **kwargs):
    # Prepend 1 to the dimension of x or y if it is 1-dimensional
    remove_x, remove_y = False, False
    if x.dim() == 1:
        x = x.view(1, x.shape[0])
        remove_x = True
    if y.dim() == 1:
        y = y.view(y.shape[0], 1)
        remove_y = True

    z = torch.matmul(x.double().data, y.double().data, *args, **kwargs)

    if remove_x:
        z.squeeze_(-2)
    if remove_y:
        z.squeeze_(-1)

    return CUDALongTensor(z)

def fix_memory_blowup():
    CUDALongTensor.matmul = matmul
    rng.generate_random_ring_element.__defaults__ = (2**23, None)