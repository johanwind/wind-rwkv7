import os
import torch as th
from torch.utils.cpp_extension import load

def fw_attn(w,q,k,v,a,b,s0):
    B,T,H,C = w.shape
    y = th.empty_like(v)
    sT = th.empty_like(s0)
    s = th.empty(B,H,T//16,C,C, dtype=th.bfloat16,device=w.device)
    th.ops.wind.forward(w,q,k,v,a,b, s0,y,s,sT)
    return y,sT,s

def bw_attn(w,q,k,v,a,b,s, dy,dsT):
    dw,dq,dk,dv,da,db,ds0 = [th.empty_like(x) for x in [w,q,k,v,a,b,dsT]]
    th.ops.wind.backward(w,q,k,v,a,b, dy,s,dsT, dw,dq,dk,dv,da,db,ds0)
    return dw,dq,dk,dv,da,db,ds0

class WindRWKV7(th.autograd.Function):
    @staticmethod
    def forward(w,q,k,v,a,b,s0):
        B,T,H,C = w.shape
        assert T%16 == 0
        if not th.compiler.is_compiling():
            assert all(i.dtype==th.bfloat16 for i in [w,q,k,v,a,b,s0])
            assert all(i.is_contiguous() for i in [w,q,k,v,a,b,s0])
            assert all(i.shape == w.shape for i in [w,q,k,v,a,b])
            assert list(s0.shape) == [B,H,C,C]
        y,sT,s = fw_attn(w,q,k,v,a,b,s0)
        return y, sT, s
    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(*inputs[:-1],output[-1])
    @staticmethod
    def backward(ctx, dy, dsT, ds):
        w,q,k,v,a,b,s = ctx.saved_tensors
        B,T,H,C = w.shape
        if dsT is None: dsT = th.zeros(B,H,C,C, dtype=th.bfloat16,device=w.device)
        if not th.compiler.is_compiling():
            assert ds is None
            assert all(i.dtype==th.bfloat16 for i in [dy,dsT])
            assert all(i.is_contiguous() for i in [dy,dsT])
        dw,dq,dk,dv,da,db,ds0 = bw_attn(w,q,k,v,a,b,s, dy,dsT)
        return dw,dq,dk,dv,da,db,ds0

def load_attn(HEAD_SIZE):
    if hasattr(th.ops.wind, 'forward'): return
    CUDA_FLAGS = ["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
    path = os.path.dirname(__file__)
    load(name="wind", sources=[os.path.join(path,'attn.cu'), os.path.join(path,'attn.cpp')], is_python_module=False, verbose=True, extra_cuda_cflags=CUDA_FLAGS+[f'-D_C_={HEAD_SIZE}'])
    assert hasattr(th.ops.wind, 'forward')

def attn(r,w,k,v,a,b, HEAD_SIZE):
    B,T,HC = w.shape
    C = HEAD_SIZE
    H = HC//C
    r,w,k,v,a,b = [i.view(B,T,H,C) for i in [r,w,k,v,a,b]]
    s0 = th.zeros(B,H,C,C, dtype=th.bfloat16,device=w.device)
    return WindRWKV7.apply(w,r,k,v,a,b,s0)[0].view(B,T,HC)
