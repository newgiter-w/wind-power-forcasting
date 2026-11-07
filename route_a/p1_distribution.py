"""Full frozen leaf-response distributions and exact discrete-law operations.

Nothing here fits a forest. ECDF semantics differ from weighted-linear QRF
quantile interpolation; the P1 protocol retains both original representations.
"""
import numpy as np
from scipy.sparse import csr_matrix,vstack
from sklearn.ensemble._forest import _generate_sample_indices


def inverse(x,w,p,boundary=False):
    p=np.asarray(p,dtype=float)
    c=np.cumsum(w,dtype=float);c/=c[-1]
    result=x[np.minimum(np.searchsorted(c,p-1e-14,side='left'),len(x)-1)]
    if boundary:
        result=np.where(p<=0,0.,np.where(p>=1,1.,result))
    return result


def distribution_stats(x,w,y,levels):
    """Exact CRPS and quantiles for a finite discrete probability law."""
    order=np.argsort(x,kind='stable');x=np.asarray(x)[order];w=np.asarray(w,dtype=float)[order]
    w=w/w.sum();cum=np.cumsum(w);prior=cum-w
    previous_mean=np.cumsum(w*x)-w*x
    crps=float(np.sum(w*np.abs(x-y))-np.sum(w*(x*prior-previous_mean)))
    assert crps>=-1e-12
    return inverse(x,w,levels),max(0.,crps)


class FrozenECDF:
    def __init__(self,forest,X_train,y_train):
        self.forest=forest
        self.support,rank=np.unique(y_train,return_inverse=True)
        n=len(y_train);blocks=[];self.lookups=[];offset=0
        for tree in forest.estimators_:
            bootstrap=_generate_sample_indices(tree.random_state,n,n)
            leaves=tree.apply(X_train)[bootstrap]
            node_ids,inv,counts=np.unique(leaves,return_inverse=True,return_counts=True)
            lookup=np.full(tree.tree_.node_count,-1,dtype='int32')
            lookup[node_ids]=offset+np.arange(len(node_ids));self.lookups.append(lookup);offset+=len(node_ids)
            mat=csr_matrix((1./counts[inv],(inv,rank[bootstrap])),shape=(len(node_ids),len(self.support)))
            mat.sum_duplicates();mat.sort_indices();blocks.append(mat)
        self.responses=vstack(blocks,format='csr')
        self.n_trees=len(forest.estimators_)

    def weights(self,X,batch=128):
        blocks=[]
        for start in range(0,len(X),batch):
            nodes=self.forest.apply(X[start:start+batch])
            rows=np.column_stack([self.lookups[t][nodes[:,t]] for t in range(self.n_trees)])
            assert (rows>=0).all()
            select=csr_matrix((np.full(rows.size,1/self.n_trees),
                (np.repeat(np.arange(len(rows)),self.n_trees),rows.ravel())),shape=(len(rows),self.responses.shape[0]))
            w=select@self.responses;w.sum_duplicates();w.sort_indices()
            np.testing.assert_allclose(np.asarray(w.sum(axis=1)).ravel(),1.,rtol=0,atol=2e-12)
            blocks.append(w)
        return vstack(blocks,format='csr')


def arrived(d,seed,score,position):
    idx=score[position];now=d['origins'][idx]
    candidates=np.r_[seed,score[:position]]
    used=candidates[d['valid'][candidates]&(d['ends'][candidates]<=now)&
        (d['ends'][candidates]>now-np.timedelta64(84,'D'))]
    assert len(used)>=40 and idx not in used
    return used


def local_pool(used,lead,state,states):
    band=np.arange((lead//6)*6,(lead//6+1)*6)
    day_grid=np.broadcast_to(used[:,None],(len(used),6))
    lead_grid=np.broadcast_to(band,(len(used),6))
    mask=states[used[:,None],band]==state
    days=day_grid[mask];leads=lead_grid[mask];n_days=len(np.unique(days))
    weight=min(.8,n_days/(n_days+20)) if n_days>=20 and len(days)>=40 else 0.
    return days,leads,weight,n_days


def mixed_values(global_values,local_values,weight):
    if weight==0:
        return np.asarray(global_values),np.full(len(global_values),1/len(global_values))
    return np.r_[global_values,local_values],np.r_[np.full(len(global_values),(1-weight)/len(global_values)),
        np.full(len(local_values),weight/len(local_values))]


def tail_quantile(values,weights,p):
    order=np.argsort(values,kind='stable')
    return float(inverse(np.asarray(values)[order],np.asarray(weights)[order],[p])[0])
