import torch as th
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def IND2(a,b,nb):
    return a*nb+b
@triton.jit
def IND3(a,b,c,nb,nc):
    return (a*nb+b)*nc+c
@triton.jit
def IND4(a,b,c,d,nb,nc,nd):
    return ((a*nb+b)*nc+c)*nd+d
@triton.jit
def tl_tanh(x):
    return (tl.sigmoid(x*2)*2-1).to(x.dtype)
@triton.jit
def tl_softplus(x):
    return tl.log(1 + tl.exp(x.to(tl.float32))).to(x.dtype)

@triton.jit
def single_lora(a_bias_, a_in_, Wa_, bi,t,i,j,jT, T:tl.constexpr,D:tl.constexpr):
    a_bias = tl.load(a_bias_+i).to(tl.float32)
    a_in = tl.load(a_in_+IND3(bi,t,j, T,D))
    Wa = tl.load(Wa_+IND2(i,jT, D))
    return tl.sigmoid(tl.dot(a_in, Wa)+a_bias), Wa, a_in

@triton.jit
def fw_triton_(w_in_,k_in_,z_in_,a_in_,ma_in_,mk_in_,Wz_,Wa_,Wma_,Wmk_,w_bias_,a_bias_,ma_bias_,mk_bias_, w_out_,k_out_,z_out_,b_out_, B:tl.constexpr,T:tl.constexpr,C:tl.constexpr,HEAD_SIZE:tl.constexpr,D:tl.constexpr, dT:tl.constexpr):
    bi = tl.program_id(2)
    hi = tl.program_id(0)
    t0 = tl.program_id(1) * dT

    t = t0+tl.arange(0,dT)[:,None]
    i = hi*HEAD_SIZE+tl.arange(0,HEAD_SIZE)[None,:]
    j = tl.arange(0,D)[None,:]
    jT = tl.arange(0,D)[:,None]

    z_in = tl.load(z_in_+IND3(bi,t,j, T,D))
    Wz = tl.load(Wz_+IND2(i,jT, D))
    z_out = tl.dot(z_in, Wz)
    tl.store(z_out_+IND3(bi,t,i, T,C), z_out.to(tl.bfloat16))

    k_in = tl.load(k_in_+IND3(bi,t,i, T,C)).to(tl.float32)
    z_in = tl.load(z_in_+IND3(bi,t,j, T,D))
    Wz = tl.load(Wz_+IND2(i,jT, D))
    z = tl.dot(tl_tanh(z_in), Wz)+k_in

    norm2 = tl.maximum(tl.sum(z*z, axis=1,keep_dims=True), 1e-12)
    z_out = z * tl.rsqrt(norm2)
    tl.store(z_out_+IND3(bi,t,i, T,C), z_out.to(tl.bfloat16))

    a = single_lora(a_bias_, a_in_, Wa_, bi,t,i,j,jT, T,D)[0]
    ma = single_lora(ma_bias_, ma_in_, Wma_, bi,t,i,j,jT, T,D)[0]
    mk = single_lora(mk_bias_, mk_in_, Wmk_, bi,t,i,j,jT, T,D)[0]

    k = k_in * (ma+a-ma*a)

    w_in = tl.load(w_in_+IND3(bi,t,i, T,C)).to(tl.float32)
    w_bias = tl.load(w_bias_+i).to(tl.float32)
    w_out = -tl_softplus(-(w_bias + w_in)) - 0.5
    tl.store(w_out_+IND3(bi,t,i, T,C), w_out.to(tl.bfloat16))
    k_out = k * tl.exp(w_out*mk)
    tl.store(k_out_+IND3(bi,t,i, T,C), k_out.to(tl.bfloat16))
    b_out = -z_out*a
    tl.store(b_out_+IND3(bi,t,i, T,C), b_out.to(tl.bfloat16))

def fw_triton(w_in, k_in, z_in, a_in, ma_in, mk_in, Wz, Wa, Wma, Wmk, w_bias, a_bias, ma_bias, mk_bias, HEAD_SIZE):
    B,T,C = w_in.shape
    w_out, k_out, z_out, b_out = [th.empty_like(w_in) for i in range(4)]
    D = Wz.shape[-1]
    dT = min(T,16)
    assert T%dT == 0
    fw_triton_[(C//HEAD_SIZE,T//dT,B)](w_in, k_in, z_in, a_in, ma_in, mk_in, Wz, Wa, Wma, Wmk, w_bias, a_bias, ma_bias, mk_bias, w_out, k_out, z_out, b_out, B,T,C,HEAD_SIZE,D, dT)
    return w_out, k_out, z_out, b_out


@triton.jit
def dsingle_lora(da,da_bias,Wa,dWa,a_in,a):
    da = da*(a-a*a)
    da_bias += tl.sum(da, axis=0,keep_dims=True)
    da = da.to(tl.bfloat16)
    da_in = tl.dot(da,Wa.trans())
    dWa = tl.dot(a_in.trans(), da, dWa)
    return da_bias, dWa, da_in

@triton.jit
def bw_triton_(w_in_,k_in_,z_in_,a_in_,ma_in_,mk_in_,Wz_,Wa_,Wma_,Wmk_,w_bias_,a_bias_,ma_bias_,mk_bias_,
               dw_in_,dk_in_,dz_in_,da_in_,dma_in_,dmk_in_,dWz_,dWa_,dWma_,dWmk_,dw_bias_,da_bias_,dma_bias_,dmk_bias_,
               dw_out_,dk_out_,dz_out_,db_out_,
               B:tl.constexpr,T:tl.constexpr,C:tl.constexpr,HEAD_SIZE:tl.constexpr,D:tl.constexpr, dT:tl.constexpr):
    H = C//HEAD_SIZE

    bi = tl.program_id(1)
    hi = tl.program_id(0)

    i = hi*HEAD_SIZE+tl.arange(0,HEAD_SIZE)[None,:]
    j = tl.arange(0,D)[None,:]
    jT = tl.arange(0,D)[:,None]

    dw_bias = tl.zeros((1,HEAD_SIZE), tl.float32)
    da_bias = tl.zeros((1,HEAD_SIZE), tl.float32)
    dma_bias = tl.zeros((1,HEAD_SIZE), tl.float32)
    dmk_bias = tl.zeros((1,HEAD_SIZE), tl.float32)
    dWa = tl.zeros((D,HEAD_SIZE), tl.float32)
    dWma = tl.zeros((D,HEAD_SIZE), tl.float32)
    dWmk = tl.zeros((D,HEAD_SIZE), tl.float32)
    dWz = tl.zeros((D,HEAD_SIZE), tl.float32)
    for t0 in range(0,T,dT):
        t = t0+tl.arange(0,dT)[:,None]

        z_in = tl.load(z_in_+IND3(bi,t,j, T,D))
        Wz = tl.load(Wz_+IND2(i,jT, D))
        z_out = tl.dot(z_in, Wz)

        k_in = tl.load(k_in_+IND3(bi,t,i, T,C)).to(tl.float32)
        z_in = tl.load(z_in_+IND3(bi,t,j, T,D))
        Wz = tl.load(Wz_+IND2(i,jT, D))
        tanh_z_in = tl_tanh(z_in)
        z = tl.dot(tanh_z_in, Wz)+k_in

        inorm = tl.rsqrt(tl.maximum(tl.sum(z*z, axis=1,keep_dims=True), 1e-12))
        z_out = z * inorm

        a,Wa,a_in = single_lora(a_bias_, a_in_, Wa_, bi,t,i,j,jT, T,D)
        ma,Wma,ma_in = single_lora(ma_bias_, ma_in_, Wma_, bi,t,i,j,jT, T,D)
        mk,Wmk,mk_in = single_lora(mk_bias_, mk_in_, Wmk_, bi,t,i,j,jT, T,D)

        k = k_in * (ma+a-ma*a)

        w_in = tl.load(w_in_+IND3(bi,t,i, T,C)).to(tl.float32)
        w_bias = tl.load(w_bias_+i).to(tl.float32)
        w_out = -tl_softplus(-(w_bias + w_in)) - 0.5
        e_w_out_mk = tl.exp(w_out*mk)
        k_out = k * e_w_out_mk
        b_out = -z_out*a

        dz_out = tl.load(dz_out_+IND3(bi,t,i, T,C)).to(tl.float32)
        db_out = tl.load(db_out_+IND3(bi,t,i, T,C)).to(tl.float32)
        dk_out = tl.load(dk_out_+IND3(bi,t,i, T,C)).to(tl.float32)
        dw_out = tl.load(dw_out_+IND3(bi,t,i, T,C)).to(tl.float32)

        dz_out += -db_out * a
        da = -db_out * z_out

        dk = dk_out * e_w_out_mk
        de_w_out_mk = dk * k
        dw_out += de_w_out_mk * mk
        dmk = de_w_out_mk * w_out

        dw_in = dw_out * tl.sigmoid(-(w_bias + w_in).to(tl.float32))
        tl.store(dw_in_+IND3(bi,t,i, T,C), dw_in.to(tl.bfloat16))
        dw_bias += tl.sum(dw_in.to(tl.float32), axis=0,keep_dims=True)

        dk_in = dk * (ma+a-a*ma)
        dma = dk * (k_in - k_in * a)
        da += dk * (k_in - k_in * ma)

        da_bias, dWa, da_in = dsingle_lora(da,da_bias,Wa,dWa,a_in,a)
        tl.store(da_in_+IND4(bi,t,hi,j, T,H,D), da_in.to(tl.bfloat16))
        dma_bias, dWma, dma_in = dsingle_lora(dma,dma_bias,Wma,dWma,ma_in,ma)
        tl.store(dma_in_+IND4(bi,t,hi,j, T,H,D), dma_in.to(tl.bfloat16))
        dmk_bias, dWmk, dmk_in = dsingle_lora(dmk,dmk_bias,Wmk,dWmk,mk_in,mk)
        tl.store(dmk_in_+IND4(bi,t,hi,j, T,H,D), dmk_in.to(tl.bfloat16))

        dinorm = tl.sum(dz_out*z, axis=1,keep_dims=True) # set to 0 when z is small (respect max(_,1e-12))?
        dz = dz_out * inorm - inorm*inorm*inorm * dinorm * z

        dk_in += dz
        tl.store(dk_in_+IND3(bi,t,i, T,C), dk_in.to(tl.bfloat16))
        dtanh_z_in = tl.dot(dz.to(tl.bfloat16), Wz.trans())
        dWz = tl.dot(tanh_z_in.trans(), dz.to(tl.bfloat16), dWz)
        dz_in = dtanh_z_in-dtanh_z_in * tanh_z_in*tanh_z_in
        tl.store(dz_in_+IND4(bi,t,hi,j, T,H,D), dz_in.to(tl.bfloat16))


    tl.atomic_add(dw_bias_+i, dw_bias.to(tl.float32))
    tl.atomic_add(da_bias_+i, da_bias.to(tl.float32))
    tl.atomic_add(dma_bias_+i, dma_bias.to(tl.float32))
    tl.atomic_add(dmk_bias_+i, dmk_bias.to(tl.float32))

    tl.atomic_add(dWa_+IND2(i,jT, D), dWa)
    tl.atomic_add(dWma_+IND2(i,jT, D), dWma)
    tl.atomic_add(dWmk_+IND2(i,jT, D), dWmk)
    tl.atomic_add(dWz_+IND2(i,jT, D), dWz)


def bw_triton(w_in,k_in,z_in,a_in,ma_in,mk_in,Wz,Wa,Wma,Wmk,w_bias,a_bias,ma_bias,mk_bias, dw_out,dk_out,dz_out,db_out, HEAD_SIZE):
    B,T,C = w_in.shape
    D = Wz.shape[-1]
    dT = min(T,16)
    assert T%dT == 0
    dw_in,dk_in = [th.empty_like(i) for i in [w_in,k_in]]
    dz_in,da_in,dma_in,dmk_in = th.empty(4,B,T,C//HEAD_SIZE,D, dtype=th.bfloat16, device=w_in.device)
    dWz,dWa,dWma,dWmk = th.zeros(4,C,D, device=w_in.device)
    dw_bias,da_bias,dma_bias,dmk_bias = th.zeros(4,C, device=w_in.device)
    bw_triton_[(C//HEAD_SIZE,B)](w_in,k_in,z_in,a_in,ma_in,mk_in,Wz,Wa,Wma,Wmk,w_bias,a_bias,ma_bias,mk_bias,
                                 dw_in,dk_in,dz_in,da_in,dma_in,dmk_in,dWz,dWa,dWma,dWmk,dw_bias,da_bias,dma_bias,dmk_bias,
                                 dw_out,dk_out,dz_out,db_out,
                                 B,T,C,HEAD_SIZE,D,dT)
    dz_in = dz_in.sum(2)
    da_in = da_in.sum(2)
    dma_in = dma_in.sum(2)
    dmk_in = dmk_in.sum(2)
    return dw_in,dk_in,dz_in,da_in,dma_in,dmk_in,dWz,dWa,dWma,dWmk,dw_bias,da_bias,dma_bias,dmk_bias


class ExpandLoras(th.autograd.Function):
    @staticmethod
    def forward(ctx, w_in,k_in,z_in,a_in,ma_in,mk_in,  Wz,Wa,Wma,Wmk,w_bias,a_bias,ma_bias,mk_bias, HEAD_SIZE):
        #print([i.is_contiguous() for i in [z_in,a_in,ma_in,mk_in]])
        z_in,a_in,ma_in,mk_in = [i.contiguous() for i in [z_in,a_in,ma_in,mk_in]] #TODO
        if not th.compiler.is_compiling():
            assert all(i.is_contiguous() for i in [w_in, k_in, z_in, a_in, ma_in, mk_in, Wz, Wa, Wma, Wmk, w_bias, a_bias, ma_bias, mk_bias])
        Wz, Wa, Wma, Wmk, w_bias, a_bias, ma_bias, mk_bias = [i.bfloat16() for i in [Wz, Wa, Wma, Wmk, w_bias, a_bias, ma_bias, mk_bias]]
        w_out, k_out, z_out, b_out = fw_triton(w_in, k_in, z_in, a_in, ma_in, mk_in, Wz, Wa, Wma, Wmk, w_bias, a_bias, ma_bias, mk_bias, HEAD_SIZE)
        ctx.save_for_backward(w_in, k_in, z_in, a_in, ma_in, mk_in, Wz, Wa, Wma, Wmk, w_bias, a_bias, ma_bias, mk_bias)
        ctx.headsz = HEAD_SIZE
        return w_out, k_out, z_out, b_out
    @staticmethod
    def backward(ctx, dw_out, dk_out, dz_out, db_out):
        w_in, k_in, z_in, a_in, ma_in, mk_in, Wz, Wa, Wma, Wmk, w_bias, a_bias, ma_bias, mk_bias = ctx.saved_tensors
        if not th.compiler.is_compiling():
            assert all(i.is_contiguous() for i in [dw_out, dk_out, dz_out, db_out])
        dw_in,dk_in,dz_in,da_in,dma_in,dmk_in,dWz,dWa,dWma,dWmk,dw_bias,da_bias,dma_bias,dmk_bias = bw_triton(w_in,k_in,z_in,a_in,ma_in,mk_in,Wz,Wa,Wma,Wmk,w_bias,a_bias,ma_bias,mk_bias, dw_out,dk_out,dz_out,db_out, ctx.headsz)
        return dw_in,dk_in,dz_in,da_in,dma_in,dmk_in,dWz,dWa,dWma,dWmk,dw_bias,da_bias,dma_bias,dmk_bias, None

expand_loras = ExpandLoras.apply
