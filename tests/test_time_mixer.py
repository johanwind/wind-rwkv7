import torch as th
F = th.nn.functional
from wind_rwkv.rwkv7 import TimeMixer
import wind_rwkv.rwkv7 as wind_rwkv7
from collections import namedtuple
from utils import *
from functorch import make_functional
from copy import deepcopy

import warnings
warnings.filterwarnings("ignore", message=".*no current CUDA context.*")

class RefTimeMixer(TimeMixer):
    def ref_attn(self, q,w,k,v,a,b, HEAD_SIZE):
        B,T,HC = q.shape
        H, C = HC//HEAD_SIZE, HEAD_SIZE
        q,w,k,v,a,b = [i.view(B,T,H,C) for i in [q,w,k,v,a,b]]
        s = th.zeros(B,H,C,C, device=q.device)
        w = th.exp(-th.exp(w))
        y = th.empty_like(v)
        for t in range(T):
            s = s * w[:,t,:,None,:] + s @ a[:,t,:,:,None] * b[:,t,:,None,:] + v[:,t,:,:,None] * k[:,t,:,None,:]
            y[:,t,:,:,None] = s @ q[:,t,:,:,None]
        return y.view(B,T,HC)

    def forward(self, x):
        B, T, C = x.size()
        H = self.n_head
        xx = F.pad(x, (0,0,1,-1)) - x

        xxx = x + xx * self.time_maa_x
        xxx = th.tanh(xxx @ self.time_maa_w1.mT).view(B*T, 4, -1).transpose(0, 1)
        xxx = (xxx @ self.time_maa_w2.mT).view(4, B, T, -1)
        xrg, xwa, xk, xv = (x + xx * (xxx+self.time_maa)).unbind(dim=0)

        r = xrg @ self.Wr.mT
        w = -F.softplus(-(self.time_decay + th.tanh(xwa @ self.time_decay_w1.mT) @ self.time_decay_w2.mT)) - 0.5
        k = xk @ self.Wk.mT
        v = xv @ self.Wv.mT # TODO: v1
        g = th.tanh(xrg @ self.gate_w1.mT) @ self.gate_w2.mT #TODO: sigmoid

        kk = k + th.tanh(xk @ self.time_kkk_w1.mT) @ self.time_kkk_w2.mT
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        a = th.sigmoid(self.time_aaaaa + (xwa @ self.time_aaa_w1.mT) @ self.time_aaa_w2.mT)

        ma = th.sigmoid(self.time_misc_a + (xwa @ self.ma_w1.mT) @ self.ma_w2.mT)
        k = k * ma + k*a * (1 - ma)
        mk = th.sigmoid(self.time_misc_k + (xk @ self.mk_w1.mT) @ self.mk_w2.mT)
        k = k * th.clamp(w*mk, max=0).exp()

        x = self.ref_attn(r, w, k, v, -kk, kk*a, C//H)
        x = F.group_norm(x.view(B*T, HC), H, self.ln_x.weight, self.ln_x.bias, eps = 64e-5).view(B,T,C)

        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.time_faaaa.view(H,-1)).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        return (x*g) @ self.Wo.mT


B = 4 # * 12
T = 1024
HC = 768
HEAD_SIZE = 64

model = TimeMixer(namedtuple('Config',['n_layer','n_head','n_embd'])(12,HC//HEAD_SIZE,HC), 6).cuda()
ref_model = RefTimeMixer(namedtuple('Config',['n_layer','n_head','n_embd'])(12,HC//HEAD_SIZE,HC), 6).cuda()

#model = th.compile(model, fullgraph=True)

params = dict(model.named_parameters())

keys = params.keys()
ref_keys = dict(ref_model.named_parameters()).keys()
x = th.randn(B, T, HC).cuda()
inputs = [i.detach().normal_(std=0.1) for i in params.values()] + [x]

def f(*inputs):
    return th.func.functional_call(model, dict(zip(keys,inputs[:-1])), (inputs[-1],))
def ref(*inputs):
    return th.func.functional_call(ref_model, dict(zip(ref_keys,inputs[:-1])), (inputs[-1],))

grad_check(f, ref, inputs, backward=True)

benchmark(f, inputs, backward=True)
