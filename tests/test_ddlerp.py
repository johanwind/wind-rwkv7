import torch as th
from wind_rwkv.rwkv7 import ddlerp
from utils import *

def ref_ddlerp(x, mix_x, Wmix1, Wmix2, mix_bias):
    B,T,C = x.shape
    dtype = x.dtype
    x, mix_x, Wmix1, Wmix2, mix_bias = [i.float() for i in [x, mix_x, Wmix1, Wmix2, mix_bias]]
    xx = th.nn.functional.pad(x, (0,0,1,-1)) - x
    xxx = x + xx * mix_x
    xxx = th.tanh(xxx @ Wmix1.mT).view(B*T, 4, -1).transpose(0, 1)
    xxx = (xxx @ Wmix2.mT).view(4, B, T, C)
    return (x + xx * (xxx+mix_bias)).to(dtype).unbind(dim=0)

def baseline(x, mix_x, Wmix1, Wmix2, mix_bias):
    with th.amp.autocast(device_type='cuda', dtype=th.bfloat16):
        B,T,C = x.shape
        dtype = x.dtype
        xx = th.nn.functional.pad(x, (0,0,1,-1)) - x
        xxx = x + xx * mix_x
        xxx = th.tanh(xxx @ Wmix1.mT).view(B*T, 4, -1).transpose(0, 1)
        xxx = (xxx @ Wmix2.mT).view(4, B, T, C)
        return (x + xx * (xxx+mix_bias)).bfloat16().unbind(dim=0)

def get_ddlerp_data(B,T,C,D):
    x,mix,W1,W2,bias = th.randn(B,T,C), th.randn(C), th.randn(D*4,C), th.randn(4,C,D), th.randn(4,1,1,C)
    x,mix,W1,W2,bias = [i.cuda()/2 for i in [x,mix,W1,W2,bias]]
    return x.bfloat16(), mix, W1, W2, bias

if __name__ == '__main__':
    B = 64
    T = 1024
    C = 768
    D = 32

    f = ddlerp
    f = th.compile(f, fullgraph=True)

    inputs = get_ddlerp_data(B,T,C,D)
    grad_check(f, ref_ddlerp, inputs, backward=True)

    #timer = FuncTimer('wind_rwkv.rwkv7.rwkv7_ddlerp', [f'fw{i}_triton' for i in [1,2]] + [f'bw{i}_triton' for i in [1,2,3,4]])
    benchmark(f, inputs, backward=True)
    #timer.print_summary()
