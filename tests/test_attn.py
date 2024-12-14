import torch as th
F = th.nn.functional
from wind_rwkv.rwkv7 import attn, load_attn, attn_triton
from utils import *

def ref_attn(q,w,k,v,a,b, HEAD_SIZE):
    dtype = w.dtype
    B,T,HC = q.shape
    H, C = HC//HEAD_SIZE, HEAD_SIZE
    q,w,k,v,a,b = [i.view(B,T,H,C) for i in [q,w,k,v,a,b]]
    s = th.zeros(B,H,C,C, device=q.device)
    q,w,k,v,a,b = [i.float() for i in [q,w,k,v,a,b]]
    w = th.exp(-th.exp(w))
    y = th.empty_like(v)
    for t in range(T):
        s = s * w[:,t,:,None,:] + s @ a[:,t,:,:,None] * b[:,t,:,None,:] + v[:,t,:,:,None] * k[:,t,:,None,:]
        y[:,t,:,:,None] = s @ q[:,t,:,:,None]
    return y.to(dtype).view(B,T,HC)

def triton_attn(r,w,k,v,a,b, HEAD_SIZE):
    B,T,C = r.shape
    H = C//HEAD_SIZE
    return attn_triton(r, w, k, v, a, b, C//H, dot_prec='fp32')

def get_attn_data(B,T,H,C):
    q,w,k,v,a,b = th.randn(6,B,T,H,C)
    w = -F.softplus(w)-0.5
    a = F.normalize(a,dim=-1)
    b = -a*th.sigmoid(b)
    return [i.bfloat16().view(B,T,H*C).cuda() for i in [q,w,k,v,a,b]]

if __name__ == '__main__':
    B1 = 2
    B2 = 64
    T = 1024
    C = 64
    H = 768 // C

    load_attn(C)
    f = attn
    #f = triton_attn # Doesn't work with benchmark()

    inputs = get_attn_data(B1,T,H,C)
    grad_check(f, ref_attn, inputs, backward=True, aux=(C,))

    inputs = get_attn_data(B2,T,H,C)
    benchmark(f, inputs, backward=True, aux=(C,))

