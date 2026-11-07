"""Synthetic optimizer/kernel tests only, never empirical forecasting results."""
import json,time
import numpy as np
from route_a.p2_models import torch,make_model,pinball,eval_loss,predict,ResidualConv


def train_fixture(name,seed=7):
    rng=np.random.default_rng(21);x=rng.normal(size=(16,24,5)).astype('float32')
    y=np.clip(.4+.15*x[:,:,0]+.1*x[:,:,1],0,1).astype('float32')
    m=make_model(name,5,y,seed);xx=torch.from_numpy(x);yy=torch.from_numpy(y)
    before=eval_loss(m,xx,yy);opt=torch.optim.AdamW(m.parameters(),lr=.005,weight_decay=.0001)
    t=time.perf_counter()
    for _ in range(30):
        m.train();opt.zero_grad(set_to_none=True);loss=pinball(m(xx),yy);loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
    after=eval_loss(m,xx,yy);out=predict(m,x)
    assert after<before and np.isfinite(out).all() and (np.diff(out,axis=-1)>=0).all()
    assert out.min()>=0 and out.max()<=1
    return out,dict(model=name,initial_loss=before,final_loss=after,optimizer_steps=30,
        seconds=time.perf_counter()-t,parameters=sum(p.numel() for p in m.parameters()))


def main():
    result=[]
    for name in ('MLP','TCN','GRU'):
        a,record=train_fixture(name);b,_=train_fixture(name)
        np.testing.assert_array_equal(a,b);record['seed_repeat_exact']=True;result.append(record)
    # The causal temporal block must not depend on later lead-token features.
    torch.manual_seed(0);block=ResidualConv(2).eval();x=torch.randn(2,24,32);changed=x.clone();changed[:,12:]+=100
    with torch.no_grad():np.testing.assert_array_equal(block(x)[:,:12],block(changed)[:,:12])
    print(json.dumps({'fixture_type':'synthetic_unit_tests_only','torch':torch.__version__,'cuda':torch.cuda.is_available(),
        'CPU_threads':torch.get_num_threads(),'causal_block_future_token_invariance':True,'checks':result}))


if __name__=='__main__':main()
