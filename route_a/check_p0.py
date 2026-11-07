"""Deterministic hand-calculated metrics and future-label invariance checks.

Artificial data here are exclusively unit-test fixtures, never empirical results.
"""
import json
import numpy as np
from route_a.p0_baseline import causal_cqr, metric_arrays, Q


def main():
    y=np.array([[0.,1.]])
    a=metric_arrays(y,np.full((1,2),.2),np.full((1,2),.8),.1)
    np.testing.assert_allclose(a['IS'],4.6)
    np.testing.assert_allclose(a['coverage'],0)
    n=70
    origins=np.datetime64('2020-01-01')+np.arange(n)*np.timedelta64(1,'D')
    yy=np.broadcast_to(np.linspace(.05,.95,n)[:,None],(n,24)).copy()
    pred=np.broadcast_to(Q,(n,24,7)).copy()
    d=dict(y=yy,origins=origins,ends=origins+np.timedelta64(24,'h'),valid=np.ones(n,bool))
    seed=np.arange(25);ids=np.arange(25,n)
    tests=[]
    for alpha in (.2,.1,.05):
        for rolling in (False,True):
            res=causal_cqr(d,pred,seed,ids,alpha,rolling)
            changed={**d,'y':yy.copy(),'valid':d['valid'].copy()}
            changed['y'][45:]=12345
            changed['valid'][45:]=False
            later=causal_cqr(changed,pred,seed,ids,alpha,rolling)
            # Issue 45 cannot use its own label or any future label/quality flag.
            for key in ('lo','hi','correction','n_cal','max_label_end'):
                np.testing.assert_array_equal(res[key][:21],later[key][:21])
            assert (res['max_label_end']<=origins[ids]).all()
            tests.append({'alpha':alpha,'rolling':rolling,'future_label_invariance':True})
    print(json.dumps({'fixture_type':'synthetic_unit_tests_only','hand_computed_IS':True,'checks':tests}))


if __name__=='__main__':main()
