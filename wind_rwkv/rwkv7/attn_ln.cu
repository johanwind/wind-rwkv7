// Copyright (c) 2024, Johan Sokrates Wind

#include "tile.cuh"
#include <assert.h>
typedef bf * __restrict__ F_;
typedef float * __restrict__ F32_;

constexpr int WARPS = _C_/16;
constexpr int fw_stages = 1, bw_stages = 1;

__device__ FTile fw_post_process(const FTile&x0, float*params, const FTile&q, const FTile&k, const FTile&v, const FTile&g, float*share) {
    constexpr int C = _C_;
    float2 mean = sumh<WARPS>(x0, (float2*)share);
    constexpr float iC = 1.f/C;
    mean.x *= iC;
    mean.y *= iC;
    FTile center;
    for (int i = 0; i < 8; i++) center.fdata[i] = x0.fdata[i]-(i/2%2?mean.y:mean.x);
    FTile sq;
    for (int i = 0; i < 8; i++) sq.fdata[i] = center.fdata[i]*center.fdata[i];
    float2 var = sumh<WARPS>(sq, (float2*)share);
    float2 istd = {rsqrtf(var.x*iC+64e-5f), rsqrtf(var.y*iC+64e-5f)};

    int j0 = threadIdx.x%4*2+threadIdx.x/32*16, j1 = j0+8;
    float ln_w[4] = {params[j0], params[j0+1], params[j1], params[j1+1]}, ln_b[4] = {params[C+j0], params[C+j0+1], params[C+j1], params[C+j1+1]};
    FTile x1;
    for (int i = 0; i < 8; i++) x1.fdata[i] = center.fdata[i]*(i/2%2?istd.y:istd.x) * ln_w[i%2+i/4*2] + ln_b[i%2+i/4*2];

    float cur_bonus[4] = {params[C*2+j0], params[C*2+j0+1], params[C*2+j1], params[C*2+j1+1]};
    FTile qk_bonus;
    for (int i = 0; i < 8; i++)
        qk_bonus.fdata[i] = q.fdata[i]*k.fdata[i]*cur_bonus[i%2+i/4*2];
    float2 dot = sumh<WARPS>(qk_bonus, (float2*)share);
    for (int i = 0; i < 8; i++) x1.fdata[i] += (i/2%2?dot.y:dot.x) * v.fdata[i];

    return x1*g;
}

//TODO: why is this (somewhat) slow?
__device__ RTile bw_post_process(const FTile&x0, float*params, const FTile&q, const FTile&k, const FTile&v, const FTile&g, const RTile&dout, 
        float*dparams, FTile&dq, FTile&dk, FTile&dv, FTile&dg, float*share) {
    constexpr int C = _C_;
    float2 mean = sumh<WARPS>(x0, (float2*)share);
    constexpr float iC = 1.f/C;
    mean.x *= iC;
    mean.y *= iC;
    FTile center;
    for (int i = 0; i < 8; i++) center.fdata[i] = x0.fdata[i]-(i/2%2?mean.y:mean.x);
    FTile sqr;
    for (int i = 0; i < 8; i++) sqr.fdata[i] = center.fdata[i]*center.fdata[i];
    float2 var = sumh<WARPS>(sqr, (float2*)share);
    float2 istd = {rsqrtf(var.x*iC+64e-5f), rsqrtf(var.y*iC+64e-5f)};

    int j0 = threadIdx.x%4*2+threadIdx.x/32*16, j1 = j0+8;
    float ln_w[4] = {params[j0], params[j0+1], params[j1], params[j1+1]}, ln_b[4] = {params[C+j0], params[C+j0+1], params[C+j1], params[C+j1+1]};
    FTile x1;
    for (int i = 0; i < 8; i++) x1.fdata[i] = center.fdata[i]*(i/2%2?istd.y:istd.x) * ln_w[i%2+i/4*2] + ln_b[i%2+i/4*2];

    float cur_bonus[4] = {params[C*2+j0], params[C*2+j0+1], params[C*2+j1], params[C*2+j1+1]};
    FTile qk_bonus;
    for (int i = 0; i < 8; i++)
        qk_bonus.fdata[i] = q.fdata[i]*k.fdata[i]*cur_bonus[i%2+i/4*2];
    float2 dot = sumh<WARPS>(qk_bonus, (float2*)share);
    for (int i = 0; i < 8; i++) x1.fdata[i] += (i/2%2?dot.y:dot.x) * v.fdata[i];

    dg = (FTile)dout * x1;
    FTile dx1 = dout * g;

    for (int i = 0; i < 8; i++) dv.fdata[i] = (i/2%2?dot.y:dot.x) * dx1.fdata[i];

    float2 ddot = sumh<WARPS>(dx1*v, (float2*)share);

    for (int i = 0; i < 8; i++) {
        float cb = cur_bonus[i%2+i/4*2], ddoti = (i/2%2?ddot.y:ddot.x);
        dq.fdata[i] = k.fdata[i]*cb*ddoti;
        dk.fdata[i] = q.fdata[i]*cb*ddoti;
        dparams[8+i%2+i/4*2] += k.fdata[i]*q.fdata[i]*ddoti;
    }

    dparams[4] += dx1.fdata[0]+dx1.fdata[2];
    dparams[5] += dx1.fdata[1]+dx1.fdata[3];
    dparams[6] += dx1.fdata[4]+dx1.fdata[6];
    dparams[7] += dx1.fdata[5]+dx1.fdata[7];

    FTile dcenter, dvar;
    for (int i = 0; i < 8; i++) {
        float istdi = (i/2%2?istd.y:istd.x), ln_wi = ln_w[i%2+i/4*2];
        dcenter.fdata[i] = dx1.fdata[i] * istdi * ln_wi;
        dparams[i%2+i/4*2] += dx1.fdata[i] * istdi * center.fdata[i];
        dvar.fdata[i] = dcenter.fdata[i] * center.fdata[i];
    }

    float2 dvar2 = sumh<WARPS>(dvar, (float2*)share);
    for (int i = 0; i < 8; i++) {
        float istdi = (i/2%2?istd.y:istd.x), dvar2i = (i/2%2?dvar2.y:dvar2.x);
        dcenter.fdata[i] -= dvar2i*iC * center.fdata[i] * istdi * istdi;
    }
    float2 dcenter_mean = sumh<WARPS>(dcenter, (float2*)share);
    for (int i = 0; i < 8; i++) {
        float dcenter_meani = (i/2%2?dcenter_mean.y:dcenter_mean.x);
        dcenter.fdata[i] -= dcenter_meani*iC;
    }
    return (RTile)dcenter;
}

__global__ void forward_kernel(int T, int H, F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_, F_ g_, F32_ params_, F_ s0_, bf* y_, bf* s_, bf* sT_) {
    constexpr int C = _C_, K = 16;
    int bi = blockIdx.y, hi = blockIdx.x;
    extern __shared__ char smem_[];
    char*smem = smem_;

    STile *sw_ = (STile*)smem; smem += sizeof(STile)*fw_stages*WARPS;
    STile *sq_ = (STile*)smem; smem += sizeof(STile)*fw_stages*WARPS;
    STile *sk_ = (STile*)smem; smem += sizeof(STile)*fw_stages*WARPS;
    STile *sv_ = (STile*)smem; smem += sizeof(STile)*fw_stages*WARPS;
    STile *sa_ = (STile*)smem; smem += sizeof(STile)*fw_stages*WARPS;
    STile *sb_ = (STile*)smem; smem += sizeof(STile)*fw_stages*WARPS;
    STile *sg_ = (STile*)smem; smem += sizeof(STile)*fw_stages*WARPS;
    char*share = (char*)smem;

    int stride = H*C;
    int warpi = threadIdx.x/32;

    auto push = [&](int t) {
        int off = bi*T*H*C + t*K*H*C + hi*C + warpi*16;
        int si = t%fw_stages;
        sw_[si*WARPS+warpi] = GTile(w_+off, stride);
        sq_[si*WARPS+warpi] = GTile(q_+off, stride);
        sk_[si*WARPS+warpi] = GTile(k_+off, stride);
        sv_[si*WARPS+warpi] = GTile(v_+off, stride);
        sa_[si*WARPS+warpi] = GTile(a_+off, stride);
        sb_[si*WARPS+warpi] = GTile(b_+off, stride);
        sg_[si*WARPS+warpi] = GTile(g_+off, stride);
    };
    for (int t = 0; t < fw_stages-1 && t < T/K; t++) push(t), __commit_group();

    FTile state[WARPS];
    for (int i = 0; i < WARPS; i++) {
        int off = bi*H*C*C + hi*C*C + warpi*16*C + i*16;
        RTile tmp;
        tmp = GTile(s0_+off, C);
        state[i] = tmp;
    }

    __shared__ float params[C*3];
    if (threadIdx.x < C) {
        params[threadIdx.x] = params_[hi*C+threadIdx.x];
        params[C+threadIdx.x] = params_[H*C+hi*C+threadIdx.x];
        params[C*2+threadIdx.x] = params_[H*C*2+hi*C+threadIdx.x];
    }

    for (int t = 0; t < T/K; t++) {
        __syncthreads();
        if (t+fw_stages-1 < T/K)
            push(t+fw_stages-1);
        __commit_group();
        __wait_groups<fw_stages-1>();
        __syncthreads();
        int si = t%fw_stages;
        STile &sw = sw_[si*WARPS+warpi], &sq = sq_[si*WARPS+warpi], &sk = sk_[si*WARPS+warpi], &sv = sv_[si*WARPS+warpi], &sa = sa_[si*WARPS+warpi], &sb = sb_[si*WARPS+warpi], &sg = sg_[si*WARPS+warpi];

        FTile w = (RTile)sw;
        apply_(w, [](float x) { return __expf(-__expf(x)); });
        FTile fw = w;
        FTile non_incl_pref = cumprodv<0,0>(fw);
        FTile incl_pref = non_incl_pref * w;
        FTile inv_incl_pref = incl_pref;
        apply_(inv_incl_pref, [](float x) { return 1.f/x; });

        RTile wq = (RTile)sq *     incl_pref, kwi = (RTile)sk * inv_incl_pref;
        RTile wa = (RTile)sa * non_incl_pref, bwi = (RTile)sb * inv_incl_pref;
        FTile ab = sum_warp<1,WARPS>((float2*)share, tril<1>(wa % bwi));
        RTile ak = sum_warp<1,WARPS>((float2*)share, tril<1>(wa % kwi));

        RTile ab_inv;
        __syncthreads();
        if (threadIdx.x < 32) ab_inv = tri_minv(ab, (float*)share);
        __syncthreads();
        ab_inv = from_warp(ab_inv, 0, (float4*)share);

        RTile vt = sv.t();
        FTile ab_ut = vt % ak;
        for (int i = 0; i < WARPS; i++)
            ab_ut += state[i] % from_warp(wa, i, (float4*)share);
        RTile ut = FTile(ab_ut % ab_inv);

        FTile y = sum_warp<1,WARPS>((float2*)share, tril<0>(wq % kwi)) % vt;
        y +=      sum_warp<1,WARPS>((float2*)share, tril<0>(wq % bwi)) % ut;
        for (int i = 0; i < WARPS; i++)
            y += from_warp(wq, i, (float4*)share) % state[i];

        y = fw_post_process(y, params, (RTile)sq, (RTile)sk, (RTile)sv, (RTile)sg, (float*)share);

        int off = bi*T*H*C + t*K*H*C + hi*C + warpi*16;
        GTile(y_+off, stride) = RTile(y);

        RTile kwt = transpose(kwi*fw), bwt = transpose(bwi*fw);
        for (int i = 0; i < WARPS; i++) {
            int off = bi*H*(T/K)*C*C + hi*(T/K)*C*C + t*C*C + warpi*16*C + i*16;
            GTile(s_+off, C) = (RTile)state[i];

            FTile fstate = state[i] * from_warp(fw, i, (float4*)share);
            fstate += vt % from_warp(kwt, i, (float4*)share);
            fstate += ut % from_warp(bwt, i, (float4*)share);
            state[i] = fstate;
        }
    }
    for (int i = 0; i < WARPS; i++) {
        int off = bi*H*C*C + hi*C*C + warpi*16*C + i*16;
        GTile(sT_+off, C) = state[i];
    }
}

void cuda_forward(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*a, bf*b, bf*g, float*params, bf*s0, bf*y, bf*s, bf*sT) {
    assert(T%16 == 0);
    constexpr int tmp_size1 = sizeof(float4)*32, tmp_size2 = sizeof(float)*16*16*2;
    constexpr int threads = 32*WARPS, shared_mem = sizeof(STile)*fw_stages*WARPS*7 + (tmp_size1 > tmp_size2 ? tmp_size1 : tmp_size2);
    static int reported = 0;
    if (!reported++) {
#if defined VERBOSE
        printf("forward_kernel() uses %d bytes of (dynamic) shared memory\n", shared_mem);
#endif
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, forward_kernel);
        int cur_mem = attr.maxDynamicSharedSizeBytes;
        if (shared_mem > cur_mem) {
#if defined VERBOSE
            printf("Increasing forward_kernel's MaxDynamicSharedMemorySize from %d to %d\n", cur_mem, shared_mem);
#endif
            assert(!cudaFuncSetAttribute(forward_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem));
        }
    }
    forward_kernel<<<dim3(H,B), dim3(threads), shared_mem>>>(T,H,w,q,k,v,a,b,g,params,s0,y,s,sT);
}


__global__ void backward_kernel(int T, int H, F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_, F_ g_, F32_ params_, F_ dy_, F_ s_, F_ dsT_, bf* dw_, bf* dq_, bf* dk_, bf* dv_, bf* da_, bf* db_, bf* dg_, float* dparams_, bf* ds0_) {
    constexpr int C = _C_, K = 16;
    int bi = blockIdx.y, hi = blockIdx.x;
    extern __shared__ char smem_[];
    char*smem = smem_;

    STile *sw_ = (STile*)smem; smem += sizeof(STile)*bw_stages*WARPS;
    STile *sq_ = (STile*)smem; smem += sizeof(STile)*bw_stages*WARPS;
    STile *sk_ = (STile*)smem; smem += sizeof(STile)*bw_stages*WARPS;
    STile *sv_ = (STile*)smem; smem += sizeof(STile)*bw_stages*WARPS;
    STile *sa_ = (STile*)smem; smem += sizeof(STile)*bw_stages*WARPS;
    STile *sb_ = (STile*)smem; smem += sizeof(STile)*bw_stages*WARPS;
    STile *sg_ = (STile*)smem; smem += sizeof(STile)*bw_stages*WARPS;
    STile *sdy_ = (STile*)smem; smem += sizeof(STile)*bw_stages*WARPS;
    STile *state_ = (STile*)smem; smem += sizeof(STile)*bw_stages*WARPS*WARPS;
    char*share = (char*)smem;

    int stride = H*C;
    int warpi = threadIdx.x/32;

    auto push = [&](int t) {
        int off = bi*T*H*C + t*K*H*C + hi*C + warpi*16;
        int si = t%bw_stages;
        sw_[si*WARPS+warpi] = GTile(w_+off, stride);
        sq_[si*WARPS+warpi] = GTile(q_+off, stride);
        sk_[si*WARPS+warpi] = GTile(k_+off, stride);
        sv_[si*WARPS+warpi] = GTile(v_+off, stride);
        sa_[si*WARPS+warpi] = GTile(a_+off, stride);
        sb_[si*WARPS+warpi] = GTile(b_+off, stride);
        sg_[si*WARPS+warpi] = GTile(g_+off, stride);
        sdy_[si*WARPS+warpi] = GTile(dy_+off, stride);
        for (int i = 0; i < WARPS; i++) {
            int off2 = bi*H*(T/K)*C*C + hi*(T/K)*C*C + t*C*C + warpi*16*C + i*16;
            state_[si*WARPS*WARPS+warpi*WARPS+i] = GTile(s_+off2, C);
        }
    };

    FTile dstate[WARPS];
    for (int i = 0; i < WARPS; i++) {
        int off = bi*H*C*C + hi*C*C + warpi*16*C + i*16;
        RTile tmp;
        tmp = GTile(dsT_+off, C);
        dstate[i] = tmp;
        __commit_group();
    }

    __shared__ float params[C*3];
    float dparams[12] = {};
    if (threadIdx.x < C) {
        params[threadIdx.x] = params_[hi*C+threadIdx.x];
        params[C+threadIdx.x] = params_[H*C+hi*C+threadIdx.x];
        params[C*2+threadIdx.x] = params_[H*C*2+hi*C+threadIdx.x];
    }

    for (int t = 0; t < bw_stages-1 && t < T/K; t++) push(T/K-1-t), __commit_group();

    for (int t = T/K-1; t >= 0; t--) {
        __syncthreads();
        if (t-bw_stages+1 >= 0)
            push(t-bw_stages+1);
        __commit_group();
        __wait_groups<bw_stages-1>();
        __syncthreads();
        int si = t%bw_stages;
        STile &sw = sw_[si*WARPS+warpi], &sq = sq_[si*WARPS+warpi], &sk = sk_[si*WARPS+warpi], &sv = sv_[si*WARPS+warpi], &sa = sa_[si*WARPS+warpi], &sb = sb_[si*WARPS+warpi], &sg = sg_[si*WARPS+warpi], &sdy = sdy_[si*WARPS+warpi];
        STile*state = state_+si*WARPS*WARPS;

        FTile w = (RTile)sw;
        apply_(w, [](float x) { return __expf(-__expf(x)); });
        FTile fw = w;
        FTile non_incl_pref = cumprodv<0,0>(fw);
        FTile incl_pref = non_incl_pref * w;
        FTile inv_incl_pref = incl_pref;
        apply_(inv_incl_pref, [](float x) { return 1.f/x; });

        RTile wq = (RTile)sq *     incl_pref, kwi = (RTile)sk * inv_incl_pref;
        RTile wa = (RTile)sa * non_incl_pref, bwi = (RTile)sb * inv_incl_pref;
        FTile ab = sum_warp<1,WARPS>((float2*)share, tril<1>(wa % bwi));
        RTile ak = sum_warp<1,WARPS>((float2*)share, tril<1>(wa % kwi));

        RTile ab_inv;
        __syncthreads();
        if (threadIdx.x < 32) ab_inv = tri_minv(ab, (float*)share);
        __syncthreads();
        ab_inv = from_warp(ab_inv, 0, (float4*)share);

        RTile vt = sv.t();
        FTile ab_ut = vt % ak;
        for (int i = 0; i < WARPS; i++)
            ab_ut += state[warpi*WARPS+i] % from_warp(wa, i, (float4*)share);
        RTile ut = FTile(ab_ut % ab_inv);

        FTile y = sum_warp<1,WARPS>((float2*)share, tril<0>(wq % kwi)) % vt;
        y +=      sum_warp<1,WARPS>((float2*)share, tril<0>(wq % bwi)) % ut;
        for (int i = 0; i < WARPS; i++)
            y += from_warp(wq, i, (float4*)share) % state[warpi*WARPS+i];

        FTile dq, dk, dv, dg;
        //dq.zero_();
        //dk.zero_();
        //dv.zero_();
        //dg.zero_();
        RTile dyt = transpose(bw_post_process(y, params, (RTile)sq, (RTile)sk, (RTile)sv, (RTile)sg, (RTile)sdy, dparams, dq, dk, dv, dg, (float*)share));
        int off = bi*T*H*C + t*K*H*C + hi*C + warpi*16;
        GTile(dg_+off, stride) = RTile(dg);

        RTile qb = sum_warp<1,WARPS>((float2*)share, tril<0>(wq % bwi));
        RTile qk = sum_warp<1,WARPS>((float2*)share, tril<0>(wq % kwi));

        FTile dut = FTile(dyt % transpose(qb));
        dv += transpose(qk) % dyt;
        for (int i = 0; i < WARPS; i++) {
            RTile dstatei = dstate[i];
            dut += dstatei % from_warp(bwi*fw, i, (float4*)share);
            dv += from_warp(kwi*fw, i, (float4*)share) % dstatei;
        }
        RTile dab_ut = FTile(dut % transpose(ab_inv));
        dv += transpose(ak) % dab_ut;

        GTile(dv_+off, stride) = RTile(dv);

        FTile dab = sum_warp<1,WARPS>((float2*)share, tril<1>(transpose(dab_ut) % transpose(ut)));
        FTile dak = sum_warp<1,WARPS>((float2*)share, tril<1>(transpose(dab_ut) % transpose(vt)));
        FTile dab_u_state0;
        dab_u_state0.zero_();
        for (int i = 0; i < WARPS; i++)
            dab_u_state0 += from_warp(transpose(dab_ut), i, (float4*)share) % state[i*WARPS+warpi].t();

        FTile da = dab_u_state0;
        da += dab % transpose(bwi);
        da += dak % transpose(kwi);
        da = non_incl_pref * da;
        GTile(da_+off, stride) = RTile(da);

        FTile dqb = sum_warp<1,WARPS>((float2*)share, tril<0>(transpose(dyt) % transpose(ut)));
        FTile dqk = sum_warp<1,WARPS>((float2*)share, tril<0>(transpose(dyt) % transpose(vt)));
        FTile dy_state0;
        dy_state0.zero_();
        for (int i = 0; i < WARPS; i++)
            dy_state0 += from_warp(transpose(dyt), i, (float4*)share) % state[i*WARPS+warpi].t();

        FTile dq2 = dy_state0;
        dq2 += dqb % transpose(bwi);
        dq2 += dqk % transpose(kwi);
        dq += incl_pref * dq2;
        GTile(dq_+off, stride) = RTile(dq);

        RTile wqt = transpose(wq), wat = transpose(wa);

        FTile u_dstate, v_dstate, dw;
        u_dstate.zero_();
        v_dstate.zero_();
        dw.zero_();
        RTile ones;
        for (int i = 0; i < 4; i++) ones.data[i] = to_bf2({1.f,1.f});
        for (int i = 0; i < WARPS; i++) {
            int tid = threadIdx.x%32;
            if (warpi == i) {
                for (int j = 0; j < WARPS; j++) {
                    RTile ra = dstate[j];
                    ((float4*)share)[j*32+tid] = *((float4*)ra.data);
                }
            }
            RTile dstatei;// = dstate[i*WARPS+warpi];
            __syncthreads();
            *((float4*)dstatei.data) = ((float4*)share)[warpi*32+tid];
            __syncthreads();
            RTile dstatei_t = transpose(dstatei);
            v_dstate += from_warp(transpose(vt), i, (float4*)share) % dstatei_t;
            u_dstate += from_warp(transpose(ut), i, (float4*)share) % dstatei_t;
            dw += ones % transpose((RTile)state[i*WARPS+warpi]*dstatei);
        }

        FTile db = fw * u_dstate;
        db += transpose(dab) % wat;
        db += transpose(dqb) % wqt;
        db = inv_incl_pref * db;
        GTile(db_+off, stride) = RTile(db);

        FTile dk2 = fw * v_dstate;
        dk2 += transpose(dak) % wat;
        dk2 += transpose(dqk) % wqt;
        dk += inv_incl_pref * dk2;
        GTile(dk_+off, stride) = RTile(dk);

        dw = fw * dw;
        dw += fast_dw<1>(dab,wa,bwi);
        dw += fast_dw<1>(dak,wa,kwi);
        dw += fast_dw<0>(dqb,wq,bwi);
        dw += fast_dw<0>(dqk,wq,kwi);
        FTile tmp;
        dw += cumsumv<0,0>(tmp = v_dstate*(fw*kwi));
        dw += cumsumv<0,0>(tmp = u_dstate*(fw*bwi));
        dw += cumsumv<0,1>(tmp = dab_u_state0*wa);
        dw += cumsumv<1,1>(tmp = dy_state0*wq);

        FTile dw_fac = (RTile)sw;
        apply_(dw_fac, [](float x) { return -__expf(x); });
        dw = dw * dw_fac;
        GTile(dw_+off, stride) = RTile(dw);

        __syncthreads();
        for (int i = 0; i < WARPS; i++) {
            FTile ndstate = dstate[i] * from_warp(fw, i, (float4*)share);
            ndstate += dyt % from_warp(wqt, i, (float4*)share);
            ndstate += dab_ut % from_warp(wat, i, (float4*)share);
            dstate[i] = ndstate;
        }
        __syncthreads();
    }
    for (int i = 0; i < WARPS; i++) {
        int off = bi*H*C*C + hi*C*C + warpi*16*C + i*16;
        GTile(ds0_+off, C) = dstate[i];
    }

    int j0 = threadIdx.x%4*2+threadIdx.x/32*16;
    for (int i = 0; i < 12; i++) {
        dparams[i] += __shfl_xor_sync(0xffffffff, dparams[i], 4);
        dparams[i] += __shfl_xor_sync(0xffffffff, dparams[i], 8);
        dparams[i] += __shfl_xor_sync(0xffffffff, dparams[i], 16);
    }
    if (threadIdx.x%32 < 4) {
        for (int i = 0; i < 4; i++) {
            int j = hi*C+j0+i%2+i/2*8;
            atomicAdd(&dparams_[j], dparams[i]);
            atomicAdd(&dparams_[H*C+j], dparams[i+4]);
            atomicAdd(&dparams_[H*C*2+j], dparams[i+8]);
        }
    }
}

void cuda_backward(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*a, bf*b, bf*g, float*params, bf*dy, bf*s, bf*dsT, bf*dw, bf*dq, bf*dk, bf*dv, bf*da, bf*db, bf*dg, float*dparams, bf*ds0) {
    assert(T%16 == 0);
    constexpr int tmp_size1 = sizeof(float4)*32*WARPS, tmp_size2 = sizeof(float)*16*16*2;
    constexpr int threads = 32*WARPS, shared_mem = sizeof(STile)*WARPS*bw_stages*(8+WARPS) + (tmp_size1 > tmp_size2 ? tmp_size1 : tmp_size2);
    static int reported = 0;
    if (!reported++) {
#if defined VERBOSE
        printf("backward_kernel() uses %d bytes of (dynamic) shared memory\n", shared_mem);
#endif
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, backward_kernel);
        int cur_mem = attr.maxDynamicSharedSizeBytes;
        if (shared_mem > cur_mem) {
#if defined VERBOSE
            printf("Increasing backward_kernel's MaxDynamicSharedMemorySize from %d to %d\n", cur_mem, shared_mem);
#endif
            assert(!cudaFuncSetAttribute(backward_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem));
        }
    }
    backward_kernel<<<dim3(H,B), dim3(threads), shared_mem>>>(T,H,w,q,k,v,a,b,g,params,dy,s,dsT,dw,dq,dk,dv,da,db,dg,dparams,ds0);
}
