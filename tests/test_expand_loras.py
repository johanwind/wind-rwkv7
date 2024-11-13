import torch as th
F = th.nn.functional
from wind_rwkv.rwkv7 import expand_loras
from utils import *

def ref_expand_loras(w_in, k_in, z_in, a_in, ma_in, mk_in, Wz, Wa, Wma, Wmk, w_bias, a_bias, ma_bias, mk_bias, HEAD_SIZE):
    w_in, k_in, z_in, a_in, ma_in, mk_in, Wz, Wa, Wma, Wmk, w_bias, a_bias, ma_bias, mk_bias = [i.float() for i in [w_in, k_in, z_in, a_in, ma_in, mk_in, Wz, Wa, Wma, Wmk, w_bias, a_bias, ma_bias, mk_bias]]
    w_out = -F.softplus(-(w_bias + w_in)) - 0.5

    z = k_in + th.tanh(z_in) @ Wz.mT
    z_out = F.normalize(z.view(B,T,-1,HEAD_SIZE), dim=-1, p=2.0).view(B,T,C)
    a = th.sigmoid( a_bias + a_in @ Wa.mT )

    ma = th.sigmoid(ma_bias + ma_in @ Wma.mT)
    k = k_in * ma + k_in*a * (1 - ma)
    mk = th.sigmoid(mk_bias + mk_in @ Wmk.mT)
    k_out = k * (w_out*mk).exp()
    b_out = -z_out*a
    return tuple(i.bfloat16() for i in [w_out, k_out, z_out, b_out])

def get_expand_loras_data(B,T,C,D):
    w_in, k_in = th.randn(2,B,T,C)
    (z_in, a_in, ma_in, mk_in) = th.randn(4,B,T,D)
    Wz, Wa, Wma, Wmk = th.randn(4,C,D)
    w_bias, a_bias, ma_bias, mk_bias = th.randn(4,C)

    inputs = [w_in, k_in, z_in, a_in, ma_in, mk_in, Wz, Wa, Wma, Wmk, w_bias, a_bias, ma_bias, mk_bias]
    inputs = [i.bfloat16().cuda() for i in inputs]
    return inputs


if __name__ == '__main__':
    B = 64
    T = 1024
    C = 768
    D = 16
    HEAD_SIZE = 64

    f = expand_loras
    f = th.compile(f, fullgraph=True)

    inputs = get_expand_loras_data(B,T,C,D)
    grad_check(f, ref_expand_loras, inputs, backward=True, aux=(HEAD_SIZE,))
    benchmark(f, inputs, backward=True, aux=(HEAD_SIZE,))
