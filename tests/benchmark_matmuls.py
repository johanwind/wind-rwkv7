import torch as th
from utils import *

def baseline(xrg,xwa,xk,xv, Wr,Wk,Wv,Wg1,Wkk,Wmk,Ww1,Wa,Wma,Wg2,Ww2):
    r = xrg @ Wr.bfloat16().mT
    g1 = xrg @ Wg1.bfloat16().mT

    k = xk @ Wk.bfloat16().mT
    kk1,mk1 = (xk @ th.cat([Wkk,Wmk]).bfloat16().mT).split([16,16],dim=2)

    v = xv @ Wv.bfloat16().mT

    w1,a1,ma1 = (xwa @ th.cat([Ww1,Wa,Wma]).bfloat16().mT).split([64,16,16],dim=2)

    g = th.tanh(g1) @ Wg2.bfloat16().mT
    w2 = th.tanh(w1) @ Ww2.bfloat16().mT
    return r, w2, k, v, kk1, a1, ma1, mk1, g


def get_data(B,T,C):
    xrg,xwa,xk,xv = [th.randn(B,T,C, dtype=th.bfloat16) for i in range(4)]
    Wr,Wk,Wv = [th.randn(C,C) for i in range(3)]
    Wg1,Wg2 = th.randn(128,C), th.randn(C,128)
    Ww1,Ww2 = th.randn(64,C), th.randn(C,64)
    Wkk,Wa,Wma,Wmk = [th.randn(16,C) for i in range(4)]
    inputs = [xrg,xwa,xk,xv, Wr,Wk,Wv,Wg1,Wkk,Wmk,Ww1,Wa,Wma,Wg2,Ww2]
    inputs = [i.cuda() for i in inputs]
    return inputs


if __name__ == '__main__':
    B = 64
    T = 1024
    C = 768

    f = baseline
    f = th.compile(f, fullgraph=True)#, mode='reduce-overhead')

    inputs = get_data(B,T,C)
    ms = benchmark(f, inputs, backward=True)
    flops = 3*C*C*6*T*B
    print(flops/1e9/ms, 'TFLOP/s')
