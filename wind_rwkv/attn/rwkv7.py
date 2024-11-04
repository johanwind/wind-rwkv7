import torch, os
from torch.utils.cpp_extension import load

class WindRWKV7(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w,q,k,v,a,b,g,params,s0):
        B,T,H,C = w.shape
        assert T%16 == 0
        assert all(i.dtype==torch.bfloat16 for i in [w,q,k,v,a,b,g,s0])
        assert params.dtype == torch.float32
        w,q,k,v,a,b,g,params,s0 = [i.contiguous() for i in [w,q,k,v,a,b,g,params,s0]]
        y = torch.empty_like(v)
        sT = torch.empty_like(s0)
        s = torch.zeros(B,H,T//16,C,C, dtype=w.dtype,device=w.device)
        torch.ops.wind.forward(w,q,k,v,a,b,g,params, s0,y,s,sT)
        ctx.save_for_backward(w,q,k,v,a,b,g,params,s)
        return y, sT
    @staticmethod
    def backward(ctx, dy, dsT):
        assert all(i.dtype==torch.bfloat16 for i in [dy,dsT])
        dy,dsT = [i.contiguous() for i in [dy,dsT]]
        w,q,k,v,a,b,g,params,s = ctx.saved_tensors
        dw,dq,dk,dv,da,db,dg,dparams,ds0 = [torch.empty_like(x) for x in [w,q,k,v,a,b,g,params,dsT]]
        dparams.zero_()
        torch.ops.wind.backward(w,q,k,v,a,b,g,params, dy,s,dsT, dw,dq,dk,dv,da,db,dg,dparams,ds0)
        return dw,dq,dk,dv,da,db,dg,dparams,ds0

def load_attn_kernel(HEAD_SIZE):
    if hasattr(torch.ops.wind, 'forward'): return
    CUDA_FLAGS = ["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
    path = os.path.dirname(__file__)
    load(name="wind", sources=[os.path.join(path,'rwkv7.cu'), os.path.join(path,'rwkv7.cpp')], is_python_module=False, verbose=True, extra_cuda_cflags=CUDA_FLAGS+[f'-D_C_={HEAD_SIZE}'])
    assert hasattr(torch.ops.wind, 'forward')

def fused_attn_ln(r,w,k,v,a,b,g, params, HEAD_SIZE):
    B,T,HC = w.shape
    C = HEAD_SIZE
    H = HC//C
    r,w,k,v,a,b,g = [i.view(B,T,H,C) for i in [r,w,k,v,a,b,g]]
    s0 = torch.zeros(B,H,C,C, dtype=torch.bfloat16,device=w.device)
    return WindRWKV7.apply(w,r,k,v,a,b,g,params,s0)[0].view(B,T,HC)
