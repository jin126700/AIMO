"""Frozen panel linear probes, provenance-gated criterion, and diagnostics."""
from __future__ import annotations
import json,hashlib
from pathlib import Path
import numpy as np
from .runtime import atomic_write_json as write

def panel_pool(z):
    z=np.asarray(z,dtype=np.float32)
    if z.ndim!=2 or not len(z) or not np.isfinite(z).all():raise ValueError('invalid or missing representations')
    # Canonical row order makes artifacts invariant even to floating point summation order.
    z=z[np.lexsort(z.T[::-1])]
    return np.concatenate([z.mean(0),z.std(0,ddof=0)])

def criterion(bounds,complete,policy):
    if not policy:return None,'unresolved'
    for k in ['source','definition_id','robust_threshold','nonrobust_threshold']:
        if k not in policy:raise ValueError('criterion needs verified source and thresholds')
    lo,hi=map(float,bounds);rt=policy['robust_threshold'];nt=policy['nonrobust_threshold']
    if not rt<nt:raise ValueError('thresholds must not overlap')
    if hi<=rt:return 1,'robust'
    if lo>=nt:return 0,'nonrobust'
    return None,'ambiguous' if complete else 'unresolved'

def labels_from_source(source,panels,policy):
    raw=json.loads((source/'labels/collection_pairs.json').read_text())
    by={}
    for x in raw:by.setdefault(x['original_id'],[]).append(x)
    result=[]
    for row in panels:
        labs=by.get(row['original_id'],[])
        drops=[x.get('signed_drop') if x.get('semantic_valid')=='verified' else None for x in labs]
        complete=len(labs)==row['coverage']['expected'] and all(x is not None for x in drops)
        lower=[x.get('drop_lower',x.get('drop_lo',-1.)) for x in labs]
        upper=[x.get('drop_upper',x.get('drop_hi',1.)) for x in labs]
        if len(labs)<row['coverage']['expected']:
            lower.append(-1.);upper.append(1.)
        if complete:bounds=[max(drops),max(drops)]
        else:bounds=[max([x if x is not None else -1. for x in lower],default=-1.),max([x if x is not None else 1. for x in upper],default=1.)]
        y,status=criterion(bounds,complete,policy)
        result.append({'original_id':row['original_id'],'split':row['split'],'label':y,'status':status,
            'criterion_missing':policy is None,'complete_outcomes':complete,'max_drop':max(drops) if complete else None,
            'max_drop_bounds':bounds,'coverage':row['coverage']})
    counts={s:{k:sum(x['split']==s and x['status']==k for x in result) for k in ['robust','nonrobust','ambiguous','unresolved']} for s in ['train','validation','known_test']}
    return {'criterion':policy,'counts':counts,'panels':result}

def shuffle_labels(labels,seed):
    ids=sorted(labels);values=np.array([labels[i] for i in ids])
    return dict(zip(ids,np.random.default_rng(seed).permutation(values).tolist()))

def metrics(y,p):
    from sklearn.metrics import balanced_accuracy_score,brier_score_loss,roc_auc_score,log_loss,precision_recall_fscore_support,confusion_matrix
    both=len(set(y))==2
    pr,re,_,_=precision_recall_fscore_support(y,p>=.5,labels=[0,1],zero_division=0)
    return {'balanced_accuracy':float(balanced_accuracy_score(y,p>=.5)) if both else None,
            'brier':float(brier_score_loss(y,p)),'auroc':float(roc_auc_score(y,p)) if both else None,
            'log_loss':float(log_loss(y,p,labels=[0,1])),'precision':pr.tolist(),'recall':re.tolist(),
            'confusion_matrix':confusion_matrix(y,p>=.5,labels=[0,1]).tolist()}

def fit_probe(x,y,v,vy,t,grid):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    if len(set(y))<2:raise ValueError('both train classes required')
    candidates=[]
    for c in grid:
        m=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=2000,random_state=0))
        m.fit(x,y);p=m.predict_proba(v)[:,1]
        candidates.append((float(np.mean((p-vy)**2)),c,m))
    _,c,m=min(candidates,key=lambda a:(a[0],a[1]))
    return m.predict_proba(t)[:,1],c

def probe_suite(panels,random,labels,spec):
    by={r['original_id']:r for r in labels['panels'] if r['label'] is not None}
    rows={r['original_id']:r for r in panels};rr={r['original_id']:r for r in random}
    ids={s:sorted(i for i,x in by.items() if x['split']==s) for s in ['train','validation','known_test']}
    ys={s:np.array([by[i]['label'] for i in ids[s]]) for s in ids}
    if any(len(set(y))<2 for y in ys.values()):
        reason='missing verified criterion' if labels['criterion'] is None else 'one or more splits lack two classes'
        return {'status':'ROBUSTNESS_PROBE_DATA_LIMIT','reason':reason,'models':{k:{'status':'NOT_FIT_DATA_LIMIT'} for k in ['constant','M0','RawChange','Random','Flow','shuffled_null']},
                'shuffle_requested':20,'shuffle_executed':0}, {'status':'NOT_ESTIMABLE_DATA_LIMIT','replicates':0}
    features={'Flow':(rows,'z'),'M0':(rows,'m0'),'RawChange':(rows,'raw'),'Random':(rr,'z')}
    results={};preds={}
    yt=ys['known_test']
    preds['constant']=np.full(len(yt),ys['train'].mean())
    for name,(data,key) in features.items():
        xs={s:np.array([data[i][key] for i in ids[s]]) for s in ids}
        pred,c=fit_probe(xs['train'],ys['train'],xs['validation'],ys['validation'],xs['known_test'],spec['probe_C'])
        preds[name]=pred;results[name]={'C':c,'metrics':metrics(yt,pred)}
    results['constant']={'metrics':metrics(yt,preds['constant'])}
    null=[]
    xs={s:np.array([rows[i]['z'] for i in ids[s]]) for s in ids}
    for seed in range(spec['shuffle_repeats']):
        shuffled=shuffle_labels(dict(zip(ids['train'],ys['train'])),seed)
        p,_=fit_probe(xs['train'],[shuffled[i] for i in ids['train']],xs['validation'],ys['validation'],xs['known_test'],spec['probe_C'])
        null.append(metrics(yt,p))
    boots={}
    rng=np.random.default_rng(0)
    for name in ['M0','RawChange','Random']:
        vals=[];ba=[]
        for _ in range(2000):
            idx=rng.integers(0,len(yt),len(yt))
            vals.append(float(np.mean((preds[name][idx]-yt[idx])**2-(preds['Flow'][idx]-yt[idx])**2)))
            if len(set(yt[idx]))==2:ba.append(metrics(yt[idx],preds['Flow'][idx])['balanced_accuracy']-metrics(yt[idx],preds[name][idx])['balanced_accuracy'])
        boots[name]={'brier_improvement_ci95':np.quantile(vals,[.025,.975]).tolist(),'balanced_accuracy_improvement_ci95':np.quantile(ba,[.025,.975]).tolist() if ba else None,'valid_balanced_accuracy_replicates':len(ba)}
    signal=all(v['brier_improvement_ci95'][0]>0 and v['balanced_accuracy_improvement_ci95'] and v['balanced_accuracy_improvement_ci95'][0]>0 for v in boots.values())
    return {'status':'ROBUSTNESS_RELEVANT_FLOW_REPRESENTATION' if signal else 'NO_ROBUSTNESS_INFORMATION_ESTABLISHED','models':results,'shuffled_null':null,'shuffle_executed':len(null),'test_originals':ids['known_test'],'predictions':{k:v.tolist() for k,v in preds.items()},'exploratory':True},boots

def finish(root,spec,panels,random,pairs):
    from .flow_representation import digest, SCHEMA
    manifest=json.loads((root/'page_manifest.json').read_text())
    source_index=json.loads((Path(spec['source'])/'pages/index.json').read_text())
    lookup={key:item['sha256'] for key,item in source_index.items()}
    checkpoint_hash=digest(root/'best_flow.pt')
    for pair in pairs:
        pair['encoder_checkpoint_sha256']=checkpoint_hash
        pair['variant_page_sha256']=lookup[pair['variant_id']]
        pair['original_page_sha256']=manifest['originals'][pair['original_id']]['pages'][0]['sha256']
        pair['representation_schema']=SCHEMA
    write(root/'pair_embeddings.json',pairs)
    artifact=json.loads((root/'representation_manifest.json').read_text())
    artifact['pair_embeddings_sha256']=digest(root/'pair_embeddings.json')
    write(root/'representation_manifest.json',artifact)
    labels=labels_from_source(Path(spec['source']),panels,spec['criterion'])
    write(root/'robustness_labels.json',labels)
    result,boot=probe_suite(panels,random,labels,spec)
    write(root/'probe_results.json',result);write(root/'bootstrap.json',boot)
    z=np.asarray([p['z'] for p in pairs])
    diag={'z_norm_quantiles':np.quantile(np.linalg.norm(z,axis=1),[0,.25,.5,.75,1]).tolist(),
          'within_original_dispersion':{p['original_id']:p['dispersion'] for p in panels},
          'robust_nonrobust_centroid_distance':None,'centroid_status':'labels unavailable'}
    by={p['original_id']:p['label'] for p in labels['panels']}
    classes={k:np.asarray([p['z'] for p in panels if by.get(p['original_id'])==k]) for k in [0,1]}
    if all(len(v) for v in classes.values()):
        diag['robust_nonrobust_centroid_distance']=float(np.linalg.norm(classes[0].mean(0)-classes[1].mean(0)))
        diag['centroid_status']='diagnostic only'
    from sklearn.decomposition import PCA
    train=np.asarray([p['z'] for p in pairs if p['split']=='train'])
    pca=PCA(n_components=2,svd_solver='full').fit(train);xy=pca.transform(z)
    diag['pca_train_fit_explained_variance_ratio']=pca.explained_variance_ratio_.tolist()
    write(root/'representation_diagnostics.json',diag)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6,4))
    for split in ['train','validation','known_test']:
        idx=[i for i,p in enumerate(pairs) if p['split']==split]
        ax.scatter(xy[idx,0],xy[idx,1],label=split,s=14,alpha=.7)
    ax.set(xlabel='PC1 (fit on train)',ylabel='PC2',title='Frozen Flow representation — diagnostic only');ax.legend()
    fig.tight_layout();fig.savefig(root/'pca_diagnostic.png',dpi=160);plt.close(fig)
