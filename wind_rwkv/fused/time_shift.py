# Copyright (c) 2024, Johan Sokrates Wind

import torch as th
import triton
import triton.language as tl

def reference(x, mix_x, Wmix1, Wmix2, mix_bias):
    B,T,C = x.shape
    dtype = x.dtype
    x, mix_x, Wmix1, Wmix2, mix_bias = [i.float() for i in [x, mix_x, Wmix1, Wmix2, mix_bias]]
    xx = th.nn.functional.pad(x, (0,0,1,-1)) - x
    xxx = x + xx * mix_x
    xxx = th.tanh(xxx @ Wmix1.mT).view(B*T, 4, -1).transpose(0, 1)
    xxx = (xxx @ Wmix2.mT).view(4, B, T, C)
    return (x + xx * (xxx+mix_bias)).to(dtype).unbind(dim=0)

@triton.jit
def IND3(a,b,c,nb,nc):
    return (a*nb+b)*nc+c
@triton.jit
def IND4(a,b,c,d,nb,nc,nd):
    return ((a*nb+b)*nc+c)*nd+d


@triton.jit
def fw1_triton_(x_, alpha_, L_, b_, T:tl.constexpr,C:tl.constexpr,D:tl.constexpr,dT:tl.constexpr,dC:tl.constexpr):
    bi = tl.program_id(0)//(T//dT)
    t0 = tl.program_id(0)%(T//dT) * dT

    acc = tl.zeros((dT,D), tl.float32)
    dt = tl.arange(0,dT)[:,None]
    i = tl.arange(0,dC)[None,:]
    j = tl.arange(0,D)[None,:]
    for i0 in range(0,C,dC):
        x = tl.load(x_+IND3(bi,t0+dt,i0+i, T,C), eviction_policy='evict_first')
        prv = tl.load(x_+IND3(bi,t0+dt-1,i0+i, T,C), mask=(t0+dt>0), eviction_policy='evict_first')
        alpha = tl.load(alpha_+i0+i)
        LT = tl.load(L_+j*C+i0+i.trans(), eviction_policy='evict_last')

        a = x + (prv-x)*alpha
        acc = tl.dot(a, LT, acc)

    acc = tl.sigmoid(acc*2)*2-1
    tl.store(b_+IND3(bi,t0+dt,j, T,D), acc.to(tl.bfloat16))

def fw1_triton(x, alpha, L):
    B,T,C = x.shape
    D = 128
    dC = min(C,32)
    dT = min(T,64)
    assert T%dT == 0
    assert C%dC == 0
    assert L.shape[0] == D
    b = th.empty((B,T,D), dtype=th.bfloat16, device=x.device)
    fw1_triton_[(B*(T//dT),)](x, alpha, L, b, T,C,D,dT,dC, num_stages=1)
    return b


@triton.jit
def fw2_triton_(x_, beta_, R_, b_, y_, B:tl.constexpr,T:tl.constexpr,C:tl.constexpr,D:tl.constexpr,dT:tl.constexpr,dC:tl.constexpr):
    bi = tl.program_id(0)//((T//dT)*(C//dC))
    t0 = tl.program_id(0)//(C//dC)%(T//dT) * dT
    i0 = tl.program_id(0)%(C//dC) * dC

    dt = tl.arange(0,dT)[:,None]
    i = tl.arange(0,dC)[None,:]
    j = tl.arange(0,D)[None,:]

    x = tl.load(x_+IND3(bi,t0+dt,i0+i, T,C))
    diff = tl.load(x_+IND3(bi,t0+dt-1,i0+i, T,C), mask=(t0+dt>0)) - x

    for k in range(4):
        beta = tl.load(beta_+k*C+i0+i)
        RT = tl.load(R_+IND3(k,i0+i,j.trans(), C,D))
        b = tl.load(b_+IND4(bi,t0+dt,k,j, T,4,D))
        c = tl.dot(b,RT) + beta
        y = x + diff * c
        tl.store(y_+IND4(k,bi,t0+dt,i0+i, B,T,C), y)

def fw2_triton(b, R, beta, x):
    B,T,C = x.shape
    D = 32
    dC = min(C,32)
    dT = min(T,64)
    assert T%dT == 0
    assert C%dC == 0
    assert list(R.shape) == [4,C,D]
    assert list(b.shape) == [B,T,4*D]
    y = th.empty((4,B,T,C), dtype=th.bfloat16, device=x.device)
    fw2_triton_[(B*(T//dT)*(C//dC),)](x, beta, R, b, y, B,T,C,D,dT,dC, num_stages=1)
    return y


@triton.jit
def bw1_triton_(dy0_,dy1_,dy2_,dy3_, x_, R_, db_,  B:tl.constexpr,T:tl.constexpr,C:tl.constexpr,D:tl.constexpr,dT:tl.constexpr,dC:tl.constexpr):
    bi = tl.program_id(0)//(T//dT)
    t0 = tl.program_id(0)%(T//dT) * dT

    t = t0 + tl.arange(0,dT)[None,:,None]
    i = tl.arange(0,dC)[None,None,:]
    iT = tl.arange(0,dC)[None,:,None]
    j = tl.arange(0,D)[None,None,:]
    k = tl.arange(0,4)[:,None,None]

    db = tl.zeros((4,dT,D), tl.float32)
    for i0 in range(0,C,dC):
        x = tl.load(x_+IND3(bi,t,i0+i, T,C), eviction_policy='evict_first')
        diff = tl.load(x_+IND3(bi,t-1,i0+i, T,C), mask=(t>0), eviction_policy='evict_first') - x

        dy = tl.zeros((4,dT,dC), tl.bfloat16)
        dy += tl.load(dy0_+IND3(bi,t,i0+i, T,C)+k*0, mask=k==0, eviction_policy='evict_first')
        dy += tl.load(dy1_+IND3(bi,t,i0+i, T,C)+k*0, mask=k==1, eviction_policy='evict_first')
        dy += tl.load(dy2_+IND3(bi,t,i0+i, T,C)+k*0, mask=k==2, eviction_policy='evict_first')
        dy += tl.load(dy3_+IND3(bi,t,i0+i, T,C)+k*0, mask=k==3, eviction_policy='evict_first')
        dc = dy * diff

        R = tl.load(R_+IND3(k,i0+iT,j, C,D), eviction_policy='evict_last')
        db = tl.dot(dc, R, db)
    tl.store(db_+IND4(bi,t,k,j, T,4,D), db.to(tl.bfloat16))

def bw1_triton(dy, x, R):
    B,T,C = x.shape
    D = 32
    dC = min(C, 128)
    dT = min(T, 32)
    assert T%dT == 0
    assert C%dC == 0
    assert list(R.shape) == [4,C,D]
    db = th.empty((B,T,4*D), dtype=th.bfloat16, device=x.device)
    bw1_triton_[(B*(T//dT),)](*dy, x, R, db, B,T,C,D,dT,dC, num_stages=1)
    return db


@triton.jit
def bw2_triton_(dy0_,dy1_,dy2_,dy3_, x_, b_, db_, L_, R_, alpha_, beta_, dx_, dL_, dR_, dalpha_, dbeta_, dx0_, B:tl.constexpr,T:tl.constexpr,C:tl.constexpr,D:tl.constexpr,dT:tl.constexpr,dC:tl.constexpr,dT2:tl.constexpr):
    bi = tl.program_id(0)//((T//dT2)*(C//dC))
    t0 = tl.program_id(0)//(C//dC)%(T//dT2) * dT2
    i0 = tl.program_id(0)%(C//dC) * dC

    dt = tl.arange(0,dT)[None,:,None]
    i = i0+tl.arange(0,dC)[None,None,:]
    j = tl.arange(0,D)[None,None,:]
    jT = tl.arange(0,D)[None,:,None]
    k = tl.arange(0,4)[:,None,None]

    dR = tl.zeros((4,dC,D), tl.float32)
    dL = tl.zeros((4*D,dC), tl.float32)
    dalpha = tl.zeros((1,1,dC), tl.float32)
    dbeta = tl.zeros((4,1,dC), tl.float32)

    for t1 in range(t0,t0+dT2,dT):
        t = t1+dt

        db = tl.load(db_+IND4(bi,t,k,j, T,4,D))
        b = tl.load(b_+IND4(bi,t,k,j, T,4,D))
        db = db - db*b*b
        db = tl.trans(db,1,0,2).reshape(dT,4*D)

        x = tl.load(x_+IND3(bi,t,i, T,C))
        diff = tl.load(x_+IND3(bi,t-1,i, T,C), mask=(t>0)) - x
        alpha = tl.load(alpha_+i)
        a = x + diff * alpha
        dL = tl.dot(db.trans(), a.reshape(dT,dC), dL)

        L = tl.load(L_+IND3(k,jT,i, D,C)).reshape(4*D,dC)
        da = tl.dot(db, L).to(tl.bfloat16)
        dalpha += tl.sum(da[None,:,:]*diff, axis=1, keep_dims=True)

        beta = tl.load(beta_+k*C+i)
        RT = tl.load(R_+IND3(k,i,jT, C,D))
        c = tl.dot(b, RT).to(tl.bfloat16) + beta

        dy = tl.zeros((4,dT,dC), tl.bfloat16)
        dy += tl.load(dy0_+IND3(bi,t,i, T,C)+k*0, mask=k==0)
        dy += tl.load(dy1_+IND3(bi,t,i, T,C)+k*0, mask=k==1)
        dy += tl.load(dy2_+IND3(bi,t,i, T,C)+k*0, mask=k==2)
        dy += tl.load(dy3_+IND3(bi,t,i, T,C)+k*0, mask=k==3)
        dc = dy * diff

        dbeta += tl.sum(dc,axis=1,keep_dims=True)
        dR = tl.dot(tl.trans(dc,0,2,1), b, dR)

        prod = tl.sum(dy*c, axis=0, keep_dims=True) + da*alpha
        dx0 = tl.sum(prod*(dt==0), axis=1, keep_dims=True)
        tl.store(dx0_+IND3(bi,t1//dT,i, T//dT,C), dx0)
        tl.store(dx_+IND3(bi,t,i, T,C), prod)
        tl.debug_barrier()
        shifted = tl.load(dx_+IND3(bi,t+1,i, T,C), mask=dt+1<dT)

        dx = tl.sum(dy, axis=0, keep_dims=True) + da - prod + shifted
        tl.store(dx_+IND3(bi,t,i, T,C), dx)

    tl.atomic_add(dL_+IND3(k,tl.trans(j,0,2,1),i, D,C), dL.reshape(4,D,dC))
    tl.atomic_add(dR_+IND3(k,tl.trans(i,0,2,1),j, C,D), dR)
    tl.atomic_add(dalpha_+i, dalpha)
    tl.atomic_add(dbeta_+k*C+i, dbeta)

def bw2_triton(dy, x, b, db, L, R, alpha, beta):
    B,T,C = x.shape
    D = 32
    dC = min(C,32)
    dT = min(T,32)
    dT2 = min(T,1024)
    assert T%dT2 == 0 and dT2%dT == 0
    assert C%dC == 0
    assert list(L.shape) == [4*D,C]
    assert list(R.shape) == [4,C,D]
    assert list(b.shape) == [B,T,4*D]
    assert list(db.shape) == [B,T,4*D]
    dx = th.empty((B,T,C), dtype=th.bfloat16, device=x.device)
    dx0 = th.empty((B,T//dT,C), dtype=th.bfloat16, device=x.device)
    dL = th.zeros((4*D,C), dtype=th.float32, device=x.device)
    dR = th.zeros((4,C,D), dtype=th.float32, device=x.device)
    dalpha = th.zeros((1,1,C), dtype=th.float32, device=x.device)
    dbeta = th.zeros((4,1,1,C), dtype=th.float32, device=x.device)
    bw2_triton_[(B*(T//dT2)*(C//dC),)](*dy,x,b,db,L,R,alpha,beta, dx,dL,dR,dalpha,dbeta, dx0, B,T,C,D,dT,dC,dT2, num_stages=1)
    dx[:,dT-1:T-1:dT,:] += dx0[:,1:,:]
    return dx, dL, dR, dalpha, dbeta

class TimeShift(th.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, L, R, beta):
        assert x.dtype == th.bfloat16
        L = L.bfloat16()
        R = R.bfloat16()
        alpha = alpha.bfloat16()
        beta = beta.bfloat16()
        b = fw1_triton(x, alpha, L)
        y = fw2_triton(b, R, beta, x)
        ctx.save_for_backward(x, alpha, L, R, beta, b)
        return y.unbind(dim=0)
    @staticmethod
    def backward(ctx, dy0, dy1, dy2, dy3):
        x, alpha, L, R, beta, b = ctx.saved_tensors
        B,T,C = x.shape
        dy = (dy0,dy1,dy2,dy3)
        db = bw1_triton(dy, x, R)
        dx, dL, dR, dalpha, dbeta = bw2_triton(dy, x, b, db, L, R, alpha, beta)
        return dx, dalpha, dL, dR, dbeta

fused_time_shift = TimeShift.apply

if __name__ == '__main__':
    from test_utils import *
    th.manual_seed(0)

    B = 4 * 12
    T = 1024
    C = 768

    x,mix,W1,W2,bias = th.randn(B,T,C), th.randn(C), th.randn(32*4,C), th.randn(4,C,32), th.randn(4,1,1,C)
    x,mix,W1,W2,bias = [i.cuda()/2 for i in [x,mix,W1,W2,bias]]
    x = x.bfloat16()

    f = wrap#TimeShift.apply
    grad_check(f, reference, (x,mix,W1,W2,bias), backward=True)

    f = th.compile(f, fullgraph=True)
    benchmark(f, (x,mix,W1,W2,bias), backward=True)
