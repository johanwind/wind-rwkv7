import torch as th
F = th.nn.functional
from wind_rwkv.rwkv7 import attn_ln, load_attn_ln, attn_triton
from utils import *

def ref_attn_ln(q,w,k,v,a,b,g, params, HEAD_SIZE):
    dtype = w.dtype
    B,T,HC = q.shape
    H, C = HC//HEAD_SIZE, HEAD_SIZE
    q,w,k,v,a,b,g = [i.view(B,T,H,C) for i in [q,w,k,v,a,b,g]]
    s = th.zeros(B,H,C,C, device=q.device)
    q,w,k,v,a,b,g = [i.float() for i in [q,w,k,v,a,b,g]]
    w = th.exp(-th.exp(w))
    y = th.empty_like(v)
    for t in range(T):
        s = s * w[:,t,:,None,:] + s @ a[:,t,:,:,None] * b[:,t,:,None,:] + v[:,t,:,:,None] * k[:,t,:,None,:]
        y[:,t,:,:,None] = s @ q[:,t,:,:,None]
    y = F.group_norm(y.view(B*T, HC), H, params[0], params[1], eps = 64e-5).view(B,T,H,C)
    y = y + (q*k*params[2].view(H,C)).sum(dim=-1, keepdim=True) * v
    y = (y * g).view(B,T,HC)
    return y.to(dtype)

def triton_attn_ln(r,w,k,v,a,b,g, params, HEAD_SIZE):
    B,T,C = r.shape
    H = C//HEAD_SIZE
    x = attn_triton(r, w, k, v, a, b, C//H, dot_prec='fp32')
    x = F.group_norm(x.float().view(B*T, C), H, params[0], params[1], eps = 64e-5).view(B,T,C)

    x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*params[2].view(H,-1)).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
    x = (x*g).bfloat16()
    return x

def get_attn_ln_data(B,T,H,C):
    q,w,k,v,a,b,g = th.randn(7,B,T,H,C)
    w = -F.softplus(w)-0.5
    a = F.normalize(a,dim=-1)
    b = -a*th.sigmoid(b)
    params = th.randn(3,H*C).cuda()
    return [i.bfloat16().view(B,T,H*C).cuda() for i in [q,w,k,v,a,b,g]]+[params]

if __name__ == '__main__':
    B1 = 2
    B2 = 64
    T = 1024
    C = 64
    H = 768 // C

    load_attn_ln(C)
    f = attn_ln
    #f = triton_attn_ln

    f = th.compile(f, fullgraph=True, mode='reduce-overhead') # reduce-overhead makes a ~40% difference!

    inputs = get_attn_ln_data(B1,T,H,C)
    grad_check(f, ref_attn_ln, inputs, backward=True, aux=(C,))

    inputs = get_attn_ln_data(B2,T,H,C)
    #timer = FuncTimer(th.ops.wind, ['forward', 'backward'])
    benchmark(f, inputs, backward=True, aux=(C,))
    #timer.print_summary()
