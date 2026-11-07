"""Synthetic unit fixtures only; not wind forecasting effectiveness evidence."""
import json
import numpy as np
from scipy.sparse import csr_matrix
from quantile_forest import RandomForestQuantileRegressor
from route_a.p1_distribution import FrozenECDF,inverse,distribution_stats,arrived
from route_a.p1_run import stage_predictions,states_from_predictions,Q,ALPHAS


def main():
    q,c=distribution_stats(np.array([0.,1.]),np.array([.5,.5]),.25,[.1,.5,.9])
    np.testing.assert_array_equal(q,[0,0,1]);np.testing.assert_allclose(c,.25)
    np.testing.assert_array_equal(inverse(np.arange(1,73),np.ones(72)/72,[.5]),[36])
    X=np.zeros((4,1));y=np.array([0,0,.5,1.])
    toy=RandomForestQuantileRegressor(n_estimators=3,min_samples_leaf=10,max_samples_leaf=None,random_state=0,n_jobs=1).fit(X,y)
    f=FrozenECDF(toy,X,y);w=f.weights(np.zeros((2,1))).toarray()
    expected=np.zeros(3)
    for bag in toy.estimators_samples_:
        for value in y[bag]:expected[np.searchsorted(f.support,value)]+=1/(3*4)
    np.testing.assert_allclose(w,np.broadcast_to(expected,w.shape),atol=1e-14)
    n=60;h=24;origins=np.datetime64('2020-01-01')+np.arange(n)*np.timedelta64(1,'D')
    yy=np.broadcast_to(np.linspace(.1,.9,n)[:,None],(n,h)).copy()
    d=dict(y=yy,origins=origins,ends=origins+np.timedelta64(1,'D'),valid=np.ones(n,bool),history=np.full((n,h),.3))
    raw=np.broadcast_to(Q,(n,h,len(Q))).copy();ecdf=raw.copy();forecast=np.arange(40,60);seed=np.arange(40);ids=forecast.copy()
    support=np.array([0.,.5,1.]);weights=csr_matrix(np.tile([.25,.5,.25],(len(forecast)*h,1)))
    pit=np.where(yy<.5,.25,np.where(yy>.5,.75,.5));states=states_from_predictions(raw,np.full((n,h),.1),d['history'])
    base={}
    for alpha in ALPHAS:
        base[f'CQR_lo_{1-alpha:.2f}']=np.full((len(ids),h),alpha/2)
        base[f'CQR_hi_{1-alpha:.2f}']=np.full((len(ids),h),1-alpha/2)
    pred,_,trace=stage_predictions(d,raw,ecdf,weights,support,pit,states,forecast,ids,seed,base)
    altered={**d,'y':yy.copy(),'valid':d['valid'].copy()};altered['y'][45:]=99;altered['valid'][45:]=False
    otherpit=pit.copy();otherpit[45:]=1
    other,_,_=stage_predictions(altered,raw,ecdf,weights,support,otherpit,states,forecast,ids,seed,base)
    np.testing.assert_array_equal(pred[:,:6],other[:,:6])
    assert (trace['max_label_end']<=origins[ids]).all()
    assert (np.diff(pred[4:6],axis=-1)>=0).all()
    print(json.dumps({'fixture_type':'synthetic_unit_tests_only','exact_discrete_CRPS':True,
        'weighted_inverse_rank_ties':True,'frozen_leaf_bootstrap_weights':True,
        'current_and_future_label_invariance_all_seven_arms':True,'full_CDF_quantiles_nested':True}))


if __name__=='__main__':main()
