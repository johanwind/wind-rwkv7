# Copyright (c) 2024, Johan Sokrates Wind

import torch as th
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
    D = L.shape[0]
    dC = min(C,32)
    dT = min(T,64)
    assert T%dT == 0
    assert C%dC == 0
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
    D = R.shape[2]
    dC = min(C,64)
    dT = min(T,64)
    assert T%dT == 0
    assert C%dC == 0
    assert list(R.shape) == [4,C,D]
    assert list(b.shape) == [B,T,4*D]
    y = th.empty((4,B,T,C), dtype=th.bfloat16, device=x.device)
    fw2_triton_[(B*(T//dT)*(C//dC),)](x, beta, R, b, y, B,T,C,D,dT,dC, num_stages=1)
    return y

@triton.jit
def bw1_store(b_,db_, db, bi,t,k,j,T,D):
    b = tl.load(b_+IND4(bi,t,k,j, T,4,D)).to(tl.float32)
    db = db-b*b*db
    tl.store(db_+IND4(bi,t,k,j, T,4,D), db.to(tl.bfloat16))

@triton.jit
def bw1_triton_(dy0_,dy1_,dy2_,dy3_, x_, R_, db_, b_,  B:tl.constexpr,T:tl.constexpr,C:tl.constexpr,D:tl.constexpr,dT:tl.constexpr,dC:tl.constexpr):
    bi = tl.program_id(0)//(T//dT)
    t0 = tl.program_id(0)%(T//dT) * dT

    t = t0 + tl.arange(0,dT)[:,None]
    i = tl.arange(0,dC)[None,:]
    iT = tl.arange(0,dC)[:,None]
    j = tl.arange(0,D)[None,:]

    db0 = tl.zeros((dT,D), tl.float32)
    db1 = tl.zeros((dT,D), tl.float32)
    db2 = tl.zeros((dT,D), tl.float32)
    db3 = tl.zeros((dT,D), tl.float32)
    for i0 in range(0,C,dC):
        x = tl.load(x_+IND3(bi,t,i0+i, T,C), eviction_policy='evict_first')
        diff = tl.load(x_+IND3(bi,t-1,i0+i, T,C), mask=(t>0), eviction_policy='evict_first') - x

        dy0 = tl.load(dy0_+IND3(bi,t,i0+i, T,C), eviction_policy='evict_first')
        dy1 = tl.load(dy1_+IND3(bi,t,i0+i, T,C), eviction_policy='evict_first')
        dy2 = tl.load(dy2_+IND3(bi,t,i0+i, T,C), eviction_policy='evict_first')
        dy3 = tl.load(dy3_+IND3(bi,t,i0+i, T,C), eviction_policy='evict_first')
        R0 = tl.load(R_+IND3(0,i0+iT,j, C,D), eviction_policy='evict_last')
        R1 = tl.load(R_+IND3(1,i0+iT,j, C,D), eviction_policy='evict_last')
        R2 = tl.load(R_+IND3(2,i0+iT,j, C,D), eviction_policy='evict_last')
        R3 = tl.load(R_+IND3(3,i0+iT,j, C,D), eviction_policy='evict_last')

        db0 = tl.dot(dy0*diff, R0, db0)
        db1 = tl.dot(dy1*diff, R1, db1)
        db2 = tl.dot(dy2*diff, R2, db2)
        db3 = tl.dot(dy3*diff, R3, db3)

    bw1_store(b_,db_, db0, bi,t,0,j,T,D)
    bw1_store(b_,db_, db1, bi,t,1,j,T,D)
    bw1_store(b_,db_, db2, bi,t,2,j,T,D)
    bw1_store(b_,db_, db3, bi,t,3,j,T,D)

def bw1_triton(dy, x, R, b):
    B,T,C = x.shape
    D = R.shape[2]
    dC = min(C, 128)
    dT = min(T, 32)
    assert T%dT == 0
    assert C%dC == 0
    assert list(R.shape) == [4,C,D]
    db = th.empty((B,T,4*D), dtype=th.bfloat16, device=x.device)
    bw1_triton_[(B*(T//dT),)](*dy, x, R, db, b, B,T,C,D,dT,dC, num_stages=1)
    return db


@triton.jit
def bw2_triton_(x_, A_, y_, m:tl.constexpr,n:tl.constexpr,k:tl.constexpr,dm:tl.constexpr,dn:tl.constexpr,dk:tl.constexpr):
    i = dm*tl.program_id(1) + tl.arange(0,dm)
    j = dn*tl.program_id(0) + tl.arange(0,dn)
    acc = tl.zeros((dm,dn), tl.float32)
    for k0 in range(0,k,dk):
        l = k0+tl.arange(0,dk)
        x = tl.load(x_+i[:,None]*k+l[None,:])
        AT = tl.load(A_+j[:,None]+l[None,:]*n, eviction_policy='evict_last')
        acc = tl.dot(x, AT.trans(), acc)
    tl.store(y_+n*i[:,None]+j[None,:], acc.to(tl.bfloat16), eviction_policy='evict_first')

def bw2_triton(x, A):
    B,T,D = x.shape
    x = x.view(B*T,D)
    m,k = x.shape
    n = A.shape[1]
    dk = min(k,64)
    dn = min(n,64)
    dm = min(m,64)
    assert m%dm == 0
    assert n%dn == 0
    assert k%dk == 0
    y = th.empty((m,n), dtype=th.bfloat16, device=x.device)
    bw2_triton_[(n//dn,m//dm)](x, A, y, m,n,k,dm,dn,dk, num_stages=2)
    return y.view(B,T,-1)



@triton.jit
def inner(beta_,b_,R_,dy_, prod,dx,diff, dbeta_k,dR_k, bi,t,i,j,k:tl.constexpr, T:tl.constexpr,C:tl.constexpr,D:tl.constexpr):
    beta = tl.load(beta_+k*C+i)
    b = tl.load(b_+IND4(bi,t,k,j, T,4,D))
    RT = tl.load(R_+IND3(k,i,j.trans(), C,D))
    c = tl.dot(b, RT).to(tl.bfloat16) + beta

    dy = tl.load(dy_+IND3(bi,t,i, T,C), eviction_policy='evict_first')
    prod += dy*c
    dx += dy

    dc = dy * diff

    dbeta_k += tl.sum(dc,axis=0,keep_dims=True)
    dR_k = tl.dot(dc.trans(), b, dR_k)
    return prod,dx,dbeta_k,dR_k 

@triton.jit
def bw3_triton_(dy0_,dy1_,dy2_,dy3_, x_, b_, da_, R_, alpha_, beta_, dx_, dR_, dalpha_, dbeta_, dx0_, B:tl.constexpr,T:tl.constexpr,C:tl.constexpr,D:tl.constexpr,dT:tl.constexpr,dC:tl.constexpr,dT2:tl.constexpr):
    bi = tl.program_id(0)//((T//dT2)*(C//dC))
    t0 = tl.program_id(0)//(C//dC)%(T//dT2) * dT2
    i0 = tl.program_id(0)%(C//dC) * dC

    dt = tl.arange(0,dT)[:,None]
    i = i0+tl.arange(0,dC)[None,:]
    j = tl.arange(0,D)[None,:]

    dR0 = tl.zeros((dC,D), tl.float32)
    dR1 = tl.zeros((dC,D), tl.float32)
    dR2 = tl.zeros((dC,D), tl.float32)
    dR3 = tl.zeros((dC,D), tl.float32)

    dalpha = tl.zeros((1,dC), tl.float32)

    dbeta0 = tl.zeros((1,dC), tl.float32)
    dbeta1 = tl.zeros((1,dC), tl.float32)
    dbeta2 = tl.zeros((1,dC), tl.float32)
    dbeta3 = tl.zeros((1,dC), tl.float32)

    for t1 in range(t0,t0+dT2,dT):
        t = t1+dt

        alpha = tl.load(alpha_+i)
        da = tl.load(da_+IND3(bi,t,i, T,C))

        x = tl.load(x_+IND3(bi,t,i, T,C))
        diff = tl.load(x_+IND3(bi,t-1,i, T,C), mask=(t>0)) - x

        dalpha += tl.sum(da*diff, axis=0, keep_dims=True)

        prod = da*alpha
        dx = da

        prod,dx,dbeta0,dR0 = inner(beta_,b_,R_,dy0_, prod,dx,diff, dbeta0,dR0, bi,t,i,j,0, T,C,D)
        prod,dx,dbeta1,dR1 = inner(beta_,b_,R_,dy1_, prod,dx,diff, dbeta1,dR1, bi,t,i,j,1, T,C,D)
        prod,dx,dbeta2,dR2 = inner(beta_,b_,R_,dy2_, prod,dx,diff, dbeta2,dR2, bi,t,i,j,2, T,C,D)
        prod,dx,dbeta3,dR3 = inner(beta_,b_,R_,dy3_, prod,dx,diff, dbeta3,dR3, bi,t,i,j,3, T,C,D)

        dx0 = tl.sum(prod*(dt==0), axis=0, keep_dims=True)
        tl.store(dx0_+IND3(bi,t1//dT,i, T//dT,C), dx0)
        tl.store(dx_+IND3(bi,t,i, T,C), prod)
        tl.debug_barrier()
        shifted = tl.load(dx_+IND3(bi,t+1,i, T,C), mask=dt+1<dT)
        dx += - prod + shifted
        tl.store(dx_+IND3(bi,t,i, T,C), dx)

    tl.atomic_add(dR_+IND3(0,i.trans(),j, C,D), dR0)
    tl.atomic_add(dR_+IND3(1,i.trans(),j, C,D), dR1)
    tl.atomic_add(dR_+IND3(2,i.trans(),j, C,D), dR2)
    tl.atomic_add(dR_+IND3(3,i.trans(),j, C,D), dR3)
    tl.atomic_add(dbeta_+0*C+i, dbeta0)
    tl.atomic_add(dbeta_+1*C+i, dbeta1)
    tl.atomic_add(dbeta_+2*C+i, dbeta2)
    tl.atomic_add(dbeta_+3*C+i, dbeta3)

    tl.atomic_add(dalpha_+i, dalpha)

def bw3_triton(dy, x, b, da, R, alpha, beta):
    B,T,C = x.shape
    D = R.shape[2]
    dC = min(C,64)
    dT = min(T,64)
    dT2 = min(T,1024)
    assert T%dT2 == 0 and dT2%dT == 0
    assert C%dC == 0
    assert list(R.shape) == [4,C,D]
    assert list(b.shape) == [B,T,4*D]
    dx = th.empty((B,T,C), dtype=th.bfloat16, device=x.device)
    dx0 = th.empty((B,T//dT,C), dtype=th.bfloat16, device=x.device)
    dR = th.zeros((4,C,D), dtype=th.float32, device=x.device)
    dalpha = th.zeros((C,), dtype=th.float32, device=x.device)
    dbeta = th.zeros((4,1,1,C), dtype=th.float32, device=x.device)
    bw3_triton_[(B*(T//dT2)*(C//dC),)](*dy,x,b,da,R,alpha,beta, dx,dR,dalpha,dbeta, dx0, B,T,C,D,dT,dC,dT2, num_stages=2)
    dx[:,dT-1:T-1:dT,:] += dx0[:,1:,:]
    return tuple(i.bfloat16() for i in [dx, dR, dalpha, dbeta])


@triton.jit
def bw4_triton_(x_, db_, alpha_, dL_, T:tl.constexpr,C:tl.constexpr,D:tl.constexpr,dT:tl.constexpr,dC:tl.constexpr,dT2:tl.constexpr):
    bi = tl.program_id(0)//((T//dT2)*(C//dC))
    t0 = tl.program_id(0)//(C//dC)%(T//dT2) * dT2
    i0 = tl.program_id(0)%(C//dC) * dC

    dt = tl.arange(0,dT)[:,None]
    i = i0+tl.arange(0,dC)[None,:]
    j = tl.arange(0,D)[None,:]

    dL = tl.zeros((D,dC), tl.float32)

    for t1 in range(t0,t0+dT2,dT):
        t = t1+dt

        db = tl.load(db_+IND3(bi,t,j, T,D))
        x = tl.load(x_+IND3(bi,t,i, T,C))
        diff = tl.load(x_+IND3(bi,t-1,i, T,C), mask=(t>0)) - x
        alpha = tl.load(alpha_+i)
        a = x + diff * alpha

        dL = tl.dot(db.trans(), a, dL)

    tl.atomic_add(dL_+IND2(j.trans(),i, C), dL)

def bw4_triton(x, db, alpha):
    B,T,C = x.shape
    D = db.shape[2]
    dC = min(C,64)
    dT = min(T,32)
    dT2 = min(T,1024)
    assert T%dT2 == 0 and dT2%dT == 0
    assert C%dC == 0
    dL = th.zeros((D,C), dtype=th.float32, device=x.device)
    bw4_triton_[(B*(T//dT2)*(C//dC),)](x, db, alpha, dL, T,C,D,dT,dC,dT2, num_stages=2)
    return dL


class DDLerp(th.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, L, R, beta):
        L,R,alpha,beta = [i.bfloat16() for i in [L,R,alpha,beta]]
        if not th.compiler.is_compiling():
            assert all(i.is_contiguous() for i in [x,alpha,L,R,beta])
            assert x.dtype == th.bfloat16
            B,T,C = x.shape
            D = R.shape[2]
            assert list(alpha.shape) == [C]
            assert list(beta.shape) == [4,1,1,C]
            assert list(L.shape) == [4*D,C]
            assert list(R.shape) == [4,C,D]
        b = fw1_triton(x, alpha, L)
        y = fw2_triton(b, R, beta, x)
        ctx.save_for_backward(x, alpha, L, R, beta, b)
        return y.unbind(dim=0)
    @staticmethod
    def backward(ctx, dy0, dy1, dy2, dy3):
        if not th.compiler.is_compiling():
            assert all(i.is_contiguous() for i in [dy0,dy1,dy2,dy3])
        x, alpha, L, R, beta, b = ctx.saved_tensors
        B,T,C = x.shape
        dy = (dy0,dy1,dy2,dy3)
        db = bw1_triton(dy, x, R, b)
        da = bw2_triton(db, L)
        dx, dR, dalpha, dbeta = bw3_triton(dy, x, b, da, R, alpha, beta)
        dL = bw4_triton(x, db, alpha)
        return dx, dalpha, dL, dR, dbeta

ddlerp = DDLerp.apply
