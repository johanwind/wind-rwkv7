import os
import torch
from torch.utils.cpp_extension import load

th = torch
F = th.nn.functional
th.set_default_device('cuda')
th.manual_seed(0)


def grad_list(params):
    r = []
    for p in params:
        if p.grad is None:
            r.append(th.zeros_like(p.data))
        else:
            r.append(p.grad.clone())
    return r

def grad_check(f1, f2, params, backward = True, dump=False, aux=()):
    params = [p.clone().requires_grad_() for p in params]
    y1 = f1(*params,*aux)
    y2 = f2(*params,*aux)
    if type(y1) != tuple: y1 = (y1,)
    if type(y2) != tuple: y2 = (y2,)
    def rel(a,b): return (a-b).norm()/max(b.norm(),1e-30)
    print('Forward error')
    for a,b in zip(y1,y2):
        if dump and rel(a,b) > 1e-4:
            print(a)
            print(b)
        print(f'{rel(a,b):.2e}  ({b.norm():.0e})')
    if not backward: return
    dy = tuple(th.randn_like(i) for i in y1)

    th.autograd.backward(y1, dy, retain_graph=True)
    d1 = grad_list(params)
    for p in params:
        if p.grad is not None:
            p.grad.random_() # So th.empty doesn't recover the gradient
        p.grad = None
    th.autograd.backward(y2, dy)
    d2 = grad_list(params)
    print('Gradient rel. errors')
    for a,b in zip(d1,d2):
        if dump and rel(a,b) > 1e-4:
            print(a)
            print(b)
        print(f'{rel(a,b):.2e}  ({b.norm():.0e})')


def naive(w,q,k,v,a,b, s0=None, return_state = False):
    dtype = w.dtype
    B,T,H,C = k.shape
    s = th.zeros(B,H,C,C) if s0 is None else s0
    w,q,k,v,a,b,s = [i.float() for i in [w,q,k,v,a,b,s]]
    w = th.exp(-th.exp(w))
    y = th.empty_like(v)
    for t in range(T):
        s = s * w[:,t,:,None,:] + s @ a[:,t,:,:,None] * b[:,t,:,None,:] + v[:,t,:,:,None] * k[:,t,:,None,:]
        y[:,t,:,:,None] = s @ q[:,t,:,:,None]
    if return_state:
        return y.to(dtype), s.to(dtype)
    else:
        return y.to(dtype)


class WindRWKV7(th.autograd.Function):
    @staticmethod
    def forward(ctx, w,q,k,v,a,b,s0):
        B,T,H,C = w.shape
        s0 = th.zeros(B,H,C,C, dtype=w.dtype,device=w.device) if s0 is None else s0
        assert T%16 == 0
        assert all(i.dtype==th.bfloat16 for i in [w,q,k,v,a,b,s0])
        assert all(i.is_contiguous() for i in [w,q,k,v,a,b,s0])
        y = th.empty_like(v)
        sT = th.empty_like(s0)
        s = th.zeros(B,H,T//16,C,C, dtype=w.dtype,device=w.device)
        th.ops.wind.forward(w,q,k,v,a,b, s0,y,s,sT)
        ctx.save_for_backward(w,q,k,v,a,b,s)
        return y, sT
    @staticmethod
    def backward(ctx, dy, dsT):
        assert all(i.dtype==th.bfloat16 for i in [dy,dsT])
        assert all(i.is_contiguous() for i in [dy,dsT])
        w,q,k,v,a,b,s = ctx.saved_tensors
        dw,dq,dk,dv,da,db,ds0 = [th.empty_like(x) for x in [w,q,k,v,a,b,dsT]]
        th.ops.wind.backward(w,q,k,v,a,b, dy,s,dsT, dw,dq,dk,dv,da,db,ds0)
        return dw,dq,dk,dv,da,db,ds0
wind_rwkv7 = WindRWKV7.apply

@th.compile
def wind(w,q,k,v,a,b, s0=None, return_state = False):
    if not 'wind' in dir(th.ops):
        B,T,H,C = w.shape
        load(name="wind", sources=['wind_rwkv7.cu', 'wind_rwkv7.cpp'], is_python_module=False, verbose=False, extra_cuda_cflags=[f'-D_C_={C}'])
    return wind_rwkv7(w,q,k,v,a,b,s0)[:return_state+1]

if __name__ == '__main__':
    B = 1
    T = 128
    C = 64
    H = 4

    f = wind
    (w,q,k,v,z),a,s0 = th.randn(5,B,T,H,C), th.rand(B,T,H)*2-1, th.randn(B,H,C,C)
    w = -F.softplus(w)-0.5
    z = F.normalize(z,dim=-1)
    w,q,k,v,z,a,s0 = [i.bfloat16() for i in [w,q,k,v,z,a,s0]]
    b = -z*a.unsqueeze(-1)
    grad_check(f, naive, (w,q,k,v,z,b,s0), backward=True, aux=(True,))
