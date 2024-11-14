import torch as th
import triton.testing

import importlib
from functools import wraps
from time import time
from collections.abc import Iterable

def grad_list(params):
    r = []
    for p in params:
        if p.grad is None:
            r.append(th.zeros_like(p.data))
        else:
            r.append(p.grad.clone())
    return r

def grad_check(f1, f2, params, backward = True, dump=False, aux=(), tol = 1e-2):
    params = [p.clone().requires_grad_() for p in params]
    y1 = f1(*params,*aux)
    y2 = f2(*params,*aux)
    if type(y1) != tuple: y1 = (y1,)
    if type(y2) != tuple: y2 = (y2,)
    def rel(a,b): return (a-b).norm()/max(b.norm(),1e-30)
    max_err = 0
    print('Forward rel. error'+'s'*(len(y1)>1))
    for a,b in zip(y1,y2):
        if dump and rel(a,b) > 1e-4:
            print(a)
            print(b)
        max_err = max(max_err, rel(a,b))
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
        max_err = max(max_err, rel(a,b))
        print(f'{rel(a,b):.2e}  ({b.norm():.0e})')

    if max_err > tol:
        print()
        print(f'Large grad_check error: {max_err*100:.1f}%!')
        print()
    return max_err

def benchmark(f, params, backward = True, aux=()):
    params = [p.clone().requires_grad_() for p in params]
    dy = None
    def wrap():
        y = f(*params,*aux)
        if not backward: return
        if not type(y) == tuple: y = (y,)
        nonlocal dy
        if dy is None: dy = tuple(th.randn_like(i) for i in y)
        th.autograd.backward(y, dy)

    ms, min_ms, max_ms = triton.testing.do_bench(wrap, quantiles=[0.5,0.2,0.8], warmup=500,rep=1000)
    print(f'{ms:.2f} ms ({min_ms:.2f} - {max_ms:.2f})')
    return ms

def flatten(x):
    return sum(map(flatten, x), []) if type(x) in [tuple, list] else [x]

class FuncTimer:
    def __init__(self, module, func_names):
        if type(module) == str:
            module = importlib.import_module(module)
        for fn in func_names:
            setattr(module, fn, self.wrap_func(getattr(module,fn)))
        self.data = {}

    def wrap_func(self, f):
        @wraps(f)
        def wrap(*args, **kw):
            th.cuda.synchronize()
            ts = time()
            result = f(*args, **kw)
            th.cuda.synchronize()
            te = time()
            name = f.__name__

            mem = 0
            r = result if type(result) == tuple else (result,)
            for i in flatten(args)+flatten(r):
                if type(i) == th.Tensor:
                    mem += i.numel()*i.element_size()
                elif not type(i) in [float,int,bool,str,type(None)]:
                    print('Warning: FuncTimer found non-tensor argument or output of type', type(i), 'in ', name)

            if not name in self.data or mem != self.data[name][0]:
                self.data[name] = [mem, []]

            self.data[name][1].append(te-ts)
            return result
        return wrap

    def print_summary(self):
        its = min(len(i[1]) for i in self.data.values())
        sum_t = 0
        for name in self.data:
            times = self.data[name][1]
            assert len(times)%its == 0
            cnt = len(times)//its
            mem = self.data[name][0] * cnt
            t = times[len(times)//2:]
            t = sum(t)/len(t) * cnt
            band = mem/2**30/max(t,1e-12)
            print('%6.1f µs %5.1f MB (%5.1f GB/s) for function %s' % (t*1e6, mem/2**20, band, name))
            sum_t += t
        print('%.1f ms accounted for' % (sum_t*1e3))
