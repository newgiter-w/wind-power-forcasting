"""Small same-information lead-sequence models with a boundary-capable monotone head."""
import os,sys,sysconfig,importlib.util
from pathlib import Path
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_name]='2'
# The inherited root contains a project statistics.py; Torch needs the stdlib.
_statistics=Path(sysconfig.get_path('stdlib'))/'statistics.py'
if 'statistics' not in sys.modules or Path(sys.modules['statistics'].__file__).resolve()!=_statistics.resolve():
    _spec=importlib.util.spec_from_file_location('statistics',_statistics)
    _mod=importlib.util.module_from_spec(_spec);sys.modules['statistics']=_mod;_spec.loader.exec_module(_mod)
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

LEVELS=(.025,.05,.1,.5,.9,.95,.975)


def configure(seed):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(int(seed));np.random.seed(int(seed))


class MonotoneHead(nn.Module):
    def __init__(self,width,initial_quantiles):
        super().__init__();self.linear=nn.Linear(width,7)
        q=np.asarray(initial_quantiles,dtype=float).copy()
        base=q[0] if q[0]>0 else -.005
        gap=np.maximum(np.diff(np.r_[base,q[1:]]),.001)
        bias=np.r_[base,np.log(np.expm1(gap/.1))]
        nn.init.normal_(self.linear.weight,std=.001)
        with torch.no_grad():self.linear.bias.copy_(torch.tensor(bias,dtype=torch.float32))

    def forward(self,x):
        z=self.linear(x)
        q=torch.cat([z[...,:1],z[...,:1]+torch.cumsum(.1*F.softplus(z[...,1:]),dim=-1)],dim=-1)
        return torch.clamp(q,0.,1.)


class MLP(nn.Module):
    def __init__(self,features,initial):
        super().__init__();self.body=nn.Sequential(nn.Linear(features,96),nn.GELU(),nn.Dropout(.1),
            nn.Linear(96,32),nn.GELU(),nn.Dropout(.1));self.head=MonotoneHead(32,initial)

    def forward(self,x):return self.head(self.body(x))


class ResidualConv(nn.Module):
    def __init__(self,dilation):
        super().__init__();self.dilation=dilation;self.conv=nn.Conv1d(32,32,3,dilation=dilation)
        self.norm=nn.LayerNorm(32);self.dropout=nn.Dropout(.1)

    def forward(self,x):
        y=self.conv(F.pad(x.transpose(1,2),(2*self.dilation,0))).transpose(1,2)
        return self.norm(x+self.dropout(F.gelu(y)))


class TCN(nn.Module):
    def __init__(self,features,initial):
        super().__init__();self.project=nn.Linear(features,32)
        self.blocks=nn.Sequential(*[ResidualConv(d) for d in (1,2,4,8)])
        self.head=MonotoneHead(32,initial)

    def forward(self,x):return self.head(self.blocks(F.gelu(self.project(x))))


class GRU(nn.Module):
    def __init__(self,features,initial):
        super().__init__();self.project=nn.Linear(features,32);self.gru=nn.GRU(32,48,batch_first=True)
        self.dropout=nn.Dropout(.1);self.norm=nn.LayerNorm(48);self.head=MonotoneHead(48,initial)

    def forward(self,x):
        y,_=self.gru(self.dropout(F.gelu(self.project(x))))
        return self.head(self.dropout(self.norm(y)))


def make_model(name,features,y,seed):
    configure(seed)
    initial=np.quantile(np.asarray(y).ravel(),LEVELS)
    return {'MLP':MLP,'TCN':TCN,'GRU':GRU}[name](features,initial)


def pinball(prediction,target):
    level=prediction.new_tensor(LEVELS);error=target.unsqueeze(-1)-prediction
    return torch.maximum(level*error,(level-1)*error).mean()


def eval_loss(model,x,y):
    model.eval()
    with torch.no_grad():return float(pinball(model(x),y))


def predict(model,x,batch=128):
    model.eval();result=[]
    with torch.no_grad():
        for start in range(0,len(x),batch):result.append(model(torch.from_numpy(x[start:start+batch])).numpy())
    return np.concatenate(result).astype('float64')
